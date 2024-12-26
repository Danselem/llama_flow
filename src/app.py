import os
from typing import List, Optional
from pathlib import Path
import pandas as pd
import asyncio

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.llms import LLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.prompts import ChatPromptTemplate
from llama_parse import LlamaParse
from llama_index.llms.groq import Groq
from llama_index.embeddings.gemini import GeminiEmbedding
# from llama_index.utils.workflow import draw_all_possible_flows

from utils import set_env
from utils import LogEvent, EnrichedInvoice, EnrichedEvent, InvoiceEvent, InvoiceOutput
from utils import EnrichedLineItem, ProductCatalog

# Setting Env Variables
set_env("LLAMA_CLOUD_API_KEY")
set_env("GROQ_API_KEY")
set_env("GOOGLE_API_KEY")

model_name = "models/embedding-001"
embed_model = GeminiEmbedding(
    model_name=model_name, title="this is a document"
)


# Extract Prompt Template
EXTRACT_PROMPT = """
Extract the invoice table data from the following content into a structured format.
Each row should include invoice_id, line_item, quantity, unit_price, and date fields.

invoice_data:
{invoice_data}

Extract each row and format according to the provided schema.
Ensure dates are in YYYY-MM-DD format and all numbers are properly formatted.
"""

def create_product_catalog_index(
    catalog_csv_path: str,
    llm: Optional[LLM] = None
) -> BaseRetriever:
    """Create a vector store index from the product catalog CSV."""
    # Read the CSV
    df = pd.read_csv(catalog_csv_path)
    
    # Create documents for indexing
    documents = []
    for _, row in df.iterrows():
        # Combine fields for embedding
        text = f"{row['standard_name']} {row['manufacturer']} {row['description']}"
        
        # Store other fields as metadata
        metadata = {
            "sku": row["sku"],
            "standard_name": row["standard_name"],
            "category": row["category"],
            "manufacturer": row["manufacturer"],
            "unit_price": row["unit_price"]
        }
        
        doc = Document(text=text, metadata=metadata)
        documents.append(doc)
    
    # Create and return the index
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model
    )
    
    return index.as_retriever(similarity_top_k=1)

class SKUMatchingWorkflow(Workflow):
    """End-to-end workflow for invoice processing and SKU matching."""

    def __init__(
        self,
        parser: LlamaParse,
        catalog_retriever: BaseRetriever,
        llm: LLM,
        output_dir: str = "data_out",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.parser = parser
        self.retriever = catalog_retriever
        self.llm = llm
        
        # Setup output directory
        out_path = Path(output_dir) / "workflow_output"
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = out_path

    @step
    async def parse_invoice(
        self, ctx: Context, ev: StartEvent
    ) -> InvoiceEvent:
        """Parse the invoice PDF and extract structured data."""
        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=">> Parsing invoice PDF"))
            
        # Parse PDF using LlamaParse
        docs = await self.parser.aload_data(ev.invoice_path)
        invoice_data = "\n".join([d.get_content(metadata_mode="all") for d in docs])
        
        # Create extraction prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that extracts structured data from invoice tables."),
            ("user", EXTRACT_PROMPT)
        ])
        
        # Extract structured data
        invoice_structured_data = await self.llm.astructured_predict(
            InvoiceOutput,
            prompt,
            invoice_data=invoice_data
        )
        
        if self._verbose:
            ctx.write_event_to_stream(
                LogEvent(msg=f">> Extracted {len(invoice_structured_data.line_items)} line items")
            )
            
        return InvoiceEvent(invoice_data=invoice_structured_data)

    @step
    async def match_skus(
        self, ctx: Context, ev: InvoiceEvent
    ) -> EnrichedEvent:
        """Match each line item with product catalog SKUs."""
        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=">> Matching SKUs"))
            
        enriched_items = []
        
        for item in ev.invoice_data.line_items:
            # Query the catalog index
            matches = self.retriever.retrieve(item.line_item)
            
            if matches and len(matches) > 0:
                top_match = matches[0]
                metadata = top_match.metadata
                enriched_item = EnrichedLineItem(
                    invoice_id=item.invoice_id,
                    original_line_item=item.line_item,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    date_=item.date_,
                    matched_sku=metadata.get("sku"),
                    standard_name=metadata.get("standard_name"),
                    category=metadata.get("category"),
                    manufacturer=metadata.get("manufacturer"),
                    match_confidence=top_match.score if hasattr(top_match, "score") else None
                )
            else:
                enriched_item = EnrichedLineItem(
                    invoice_id=item.invoice_id,
                    original_line_item=item.line_item,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    date_=item.date_
                )
            
            enriched_items.append(enriched_item)

        return EnrichedEvent(
            enriched_data=EnrichedInvoice(line_items=enriched_items)
        )

    @step
    async def save_output(
        self, ctx: Context, ev: EnrichedEvent
    ) -> StopEvent:
        """Save the enriched data to CSV."""
        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=">> Saving enriched data"))
            
        # Convert to DataFrame
        output_data = []
        for item in ev.enriched_data.line_items:
            output_data.append({
                "invoice_id": item.invoice_id,
                "line_item": item.original_line_item,
                "quantity": item.quantity,
                "unit_price": float(item.unit_price),
                "date": item.date_,
                "matched_sku": item.matched_sku,
                "standard_name": item.standard_name,
                "category": item.category,
                "manufacturer": item.manufacturer,
                "match_confidence": item.match_confidence
            })
            
        df = pd.DataFrame(output_data)
        
        # Save to CSV
        output_path = self.output_dir / "enriched_invoice.csv"
        df.to_csv(output_path, index=False)
        
        if self._verbose:
            ctx.write_event_to_stream(
                LogEvent(msg=f">> Saved output to {output_path}")
            )
        
        return StopEvent(result=ev.enriched_data)
    


parser = LlamaParse(result_type="markdown")
llm = Groq(model="llama3-70b-8192",)

# Create product catalog index
catalog_retriever = create_product_catalog_index(
    "data/product-catalog.csv",
    llm=llm
)

# Initialize workflow
workflow = SKUMatchingWorkflow(
    parser=parser,
    catalog_retriever=catalog_retriever,
    llm=llm,
    verbose=True,
    timeout=300
)


# draw_all_possible_flows(SKUMatchingWorkflow, filename="sku_matching_workflow.html")

# Run the workflow

async def main():
    handler = workflow.run(invoice_path="invoice.pdf")
    
    # Asynchronously iterate over events
    async for event in handler.stream_events():
        if isinstance(event, LogEvent):
            print(event.msg)
    
    # Await the result of the handler
    result = await handler
    return result

# Run the main asynchronous function
if __name__ == "__main__":
    asyncio.run(main())