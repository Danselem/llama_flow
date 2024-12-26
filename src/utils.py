import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import date

from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, Field

from llama_index.core.workflow import Event



def set_env(var: str):
    # Load environment variables from the .env file
    dotenv_path = Path('./.env')
    load_dotenv(dotenv_path=dotenv_path)

    # Retrieve the value of the environment variable
    value = os.getenv(var)

    if value is None:
        raise ValueError(f"The environment variable '{var}' is not set in the .env file or system environment.")
    
    # Set the environment variable
    os.environ[var] = value



# Pydantic Models for Invoice Data
class InvoiceLineItem(BaseModel):
    """
    Represents a single line item from an invoice.
    Each line item contains details about the product ordered, quantity, and pricing.
    """
    invoice_id: str = Field(
        description="Unique identifier for the invoice (e.g., 'INV-2024-001')"
    )
    line_item: str = Field(
        description="Description of the product or service as it appears on the invoice"
    )
    quantity: int = Field(
        description="Number of units ordered",
        gt=0  # Ensures quantity is greater than 0
    )
    unit_price: Decimal = Field(
        description="Price per unit in decimal format (e.g., 45.99)",
        gt=0  # Ensures price is greater than 0
    )
    date_: date = Field(
        description="Date of the invoice in YYYY-MM-DD format"
    )

class InvoiceOutput(BaseModel):
    """
    Container model for all line items from an invoice.
    Used as the output format when parsing invoice documents.
    """
    line_items: List[InvoiceLineItem] = Field(
        description="List of all line items extracted from the invoice"
    )

# Pydantic Models for Product Catalog
class ProductCatalog(BaseModel):
    """
    Represents a single product entry in the product catalog.
    Contains standardized product information and SKU details.
    """
    sku: str = Field(
        description="Stock Keeping Unit - unique identifier for the product (e.g., 'ACM-WX-001')"
    )
    standard_name: str = Field(
        description="Standardized product name used across the system"
    )
    category: str = Field(
        description="Product category or classification (e.g., 'Widgets', 'Fasteners')"
    )
    manufacturer: str = Field(
        description="Name of the product manufacturer or supplier"
    )
    description: str = Field(
        description="Detailed product description"
    )
    unit_price: Decimal = Field(
        description="Standard unit price in decimal format",
        gt=0
    )

# Pydantic Models for Enriched Output
class EnrichedLineItem(BaseModel):
    """
    Enhanced version of InvoiceLineItem that includes matched product catalog information.
    Combines original invoice data with standardized product details.
    """
    invoice_id: str = Field(
        description="Original invoice identifier"
    )
    original_line_item: str = Field(
        description="Original product description from the invoice"
    )
    quantity: int = Field(
        description="Quantity ordered",
        gt=0
    )
    unit_price: Decimal = Field(
        description="Original unit price from invoice",
        gt=0
    )
    date_: date = Field(
        description="Invoice date"
    )
    matched_sku: Optional[str] = Field(
        None,
        description="Matched SKU from product catalog, if found"
    )
    standard_name: Optional[str] = Field(
        None,
        description="Standardized product name from catalog"
    )
    category: Optional[str] = Field(
        None,
        description="Product category from catalog"
    )
    manufacturer: Optional[str] = Field(
        None,
        description="Manufacturer information from catalog"
    )
    match_confidence: Optional[float] = Field(
        None,
        description="Confidence score of the SKU match (0.0 to 1.0)",
        ge=0.0,  # Greater than or equal to 0
        le=1.0   # Less than or equal to 1
    )

class EnrichedInvoice(BaseModel):
    """
    Container model for enriched invoice data.
    Contains all line items with their matched product catalog information.
    """
    line_items: List[EnrichedLineItem] = Field(
        description="List of all enriched line items with matched catalog data"
    )


# Event Classes
class InvoiceEvent(Event):
    """
    Workflow event that carries parsed invoice data.
    Triggered after successful invoice parsing step.
    """
    invoice_data: InvoiceOutput = Field(
        description="Parsed invoice data containing all line items"
    )

class EnrichedEvent(Event):
    """
    Workflow event that carries enriched invoice data.
    Triggered after successful SKU matching step.
    """
    enriched_data: EnrichedInvoice = Field(
        description="Enriched invoice data with matched catalog information"
    )

class LogEvent(Event):
    """
    Workflow event for logging messages and progress updates.
    Used throughout the workflow to provide status information.
    """
    msg: str = Field(
        description="Log message content"
    )
    delta: bool = Field(
        False,
        description="Flag indicating if this is a partial update to previous message"
    )