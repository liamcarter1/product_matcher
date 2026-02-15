"""
ProductMatchPro - Data Models
Pydantic models for hydraulic products, match results, and score breakdowns.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ProductCategory(str, Enum):
    DIRECTIONAL_VALVES = "directional_valves"
    PROPORTIONAL_DIRECTIONAL_VALVES = "proportional_directional_valves"
    PRESSURE_VALVES = "pressure_valves"
    FLOW_VALVES = "flow_valves"
    PUMPS = "pumps"
    MOTORS = "motors"
    CYLINDERS = "cylinders"
    FILTERS = "filters"
    ACCUMULATORS = "accumulators"
    HOSES_FITTINGS = "hoses_fittings"
    OTHER = "other"


class DocumentType(str, Enum):
    CATALOGUE = "catalogue"
    USER_GUIDE = "user_guide"
    DATASHEET = "datasheet"


class HydraulicProduct(BaseModel):
    """Core product record stored in SQLite.
    Every field that can be extracted from a model code breakdown
    or catalogue/user guide spec table is captured here."""

    id: str
    company: str
    model_code: str
    product_name: str = ""
    category: str = ""
    subcategory: Optional[str] = None

    # Core hydraulic specs
    max_pressure_bar: Optional[float] = None
    max_flow_lpm: Optional[float] = None
    valve_size: Optional[str] = None
    spool_type: Optional[str] = None
    num_positions: Optional[int] = None
    num_ports: Optional[int] = None

    # Actuation / coil specs
    actuator_type: Optional[str] = None
    coil_voltage: Optional[str] = None
    coil_type: Optional[str] = None
    coil_connector: Optional[str] = None

    # Physical / mounting
    port_size: Optional[str] = None
    port_type: Optional[str] = None
    mounting: Optional[str] = None
    mounting_pattern: Optional[str] = None

    # Materials & seals
    body_material: Optional[str] = None
    seal_material: Optional[str] = None

    # Operating conditions
    operating_temp_min_c: Optional[float] = None
    operating_temp_max_c: Optional[float] = None
    fluid_type: Optional[str] = None
    viscosity_range_cst: Optional[str] = None

    # Physical dimensions
    weight_kg: Optional[float] = None

    # Pump/motor specific
    displacement_cc: Optional[float] = None
    speed_rpm_max: Optional[float] = None

    # Cylinder specific
    bore_diameter_mm: Optional[float] = None
    rod_diameter_mm: Optional[float] = None
    stroke_mm: Optional[float] = None

    # Dynamic overflow — stores ALL specs beyond the known fields above
    extra_specs: Optional[dict] = None

    # Raw data
    description: str = ""
    source_document: str = ""
    raw_text: str = ""
    model_code_decoded: Optional[dict] = None


# All spec fields that can be compared between products
SPEC_FIELDS = [
    "max_pressure_bar", "max_flow_lpm", "valve_size", "spool_type",
    "num_positions", "num_ports", "actuator_type", "coil_voltage",
    "coil_type", "coil_connector", "port_size", "port_type",
    "mounting", "mounting_pattern", "body_material", "seal_material",
    "operating_temp_min_c", "operating_temp_max_c", "fluid_type",
    "viscosity_range_cst", "weight_kg", "displacement_cc",
    "speed_rpm_max", "bore_diameter_mm", "rod_diameter_mm", "stroke_mm",
]

# Fields that are compared numerically (closeness score)
NUMERICAL_FIELDS = {
    "max_pressure_bar", "max_flow_lpm", "weight_kg",
    "operating_temp_min_c", "operating_temp_max_c",
    "displacement_cc", "speed_rpm_max",
    "bore_diameter_mm", "rod_diameter_mm", "stroke_mm",
    "num_positions", "num_ports",
}

# Fields that must match exactly
EXACT_MATCH_FIELDS = {
    "valve_size", "spool_type", "actuator_type", "coil_voltage",
    "coil_type", "coil_connector", "port_size", "port_type",
    "mounting", "mounting_pattern", "body_material", "seal_material",
    "fluid_type", "viscosity_range_cst",
}


class ScoreBreakdown(BaseModel):
    """Full transparency into how each spec contributed to the confidence score."""

    category_match: float = Field(default=0.0, ge=0.0, le=1.0)

    # Core hydraulic specs
    pressure_match: float = Field(default=0.0, ge=0.0, le=1.0)
    flow_match: float = Field(default=0.0, ge=0.0, le=1.0)
    valve_size_match: float = Field(default=0.0, ge=0.0, le=1.0)

    # Actuation / electrical
    actuator_type_match: float = Field(default=0.0, ge=0.0, le=1.0)
    coil_voltage_match: float = Field(default=0.0, ge=0.0, le=1.0)
    spool_function_match: float = Field(default=0.0, ge=0.0, le=1.0)

    # Physical compatibility
    port_match: float = Field(default=0.0, ge=0.0, le=1.0)
    mounting_match: float = Field(default=0.0, ge=0.0, le=1.0)

    # Materials / conditions
    seal_material_match: float = Field(default=0.0, ge=0.0, le=1.0)
    temp_range_match: float = Field(default=0.0, ge=0.0, le=1.0)

    # Semantic similarity
    semantic_similarity: float = Field(default=0.0, ge=0.0, le=1.0)

    # Coverage metric
    spec_coverage: float = Field(default=0.0, ge=0.0, le=1.0)


# Weights for confidence score calculation
SCORE_WEIGHTS = {
    "category_match": 0.10,
    "pressure_match": 0.10,
    "flow_match": 0.10,
    "valve_size_match": 0.10,
    "actuator_type_match": 0.08,
    "coil_voltage_match": 0.10,
    "spool_function_match": 0.08,
    "port_match": 0.06,
    "mounting_match": 0.08,
    "seal_material_match": 0.03,
    "temp_range_match": 0.02,
    "semantic_similarity": 0.15,
}

CONFIDENCE_THRESHOLD = 0.75


class MatchResult(BaseModel):
    """A single match result returned to the distributor."""

    my_company_product: HydraulicProduct
    competitor_product: HydraulicProduct
    confidence_score: float = Field(ge=0.0, le=1.0)
    score_breakdown: ScoreBreakdown
    comparison_notes: str = ""
    meets_threshold: bool = False


class ExtractedProduct(BaseModel):
    """A product extracted from a PDF during ingestion, before confirmation."""

    model_code: str
    product_name: str = ""
    category: str = ""
    specs: dict = Field(default_factory=dict)
    raw_text: str = ""
    page_number: Optional[int] = None
    confidence: float = Field(default=0.0, description="Extraction confidence")
    source: str = "llm"  # "table", "llm", or "ordering_code"


class OrderingCodeSegment(BaseModel):
    """One position in an ordering code breakdown table."""

    position: int
    segment_name: str
    is_fixed: bool = True
    separator_before: str = ""
    options: list[dict] = Field(default_factory=list)


class OrderingCodeDefinition(BaseModel):
    """Full ordering code breakdown extracted from a datasheet.

    Example: A Bosch Rexroth 4WRE ordering code table has fixed segments
    (series, size) and variable segments (flow rate, seal material, interface).
    The combinatorial generator uses this to produce all valid product variants.
    """

    company: str
    series: str
    product_name: str = ""
    category: str = ""
    code_template: str = ""
    segments: list[OrderingCodeSegment] = Field(default_factory=list)
    shared_specs: dict = Field(default_factory=dict)


class ModelCodePattern(BaseModel):
    """A single decode rule for a model code segment."""

    company: str
    series: str
    segment_position: int
    segment_name: str
    code_value: str
    decoded_value: str
    maps_to_field: str


class UploadMetadata(BaseModel):
    """Metadata provided by admin when uploading a document."""

    company: str
    document_type: DocumentType
    category: Optional[str] = None
    filename: str = ""
