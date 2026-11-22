from datetime import date
from decimal import Decimal
from typing import Annotated, Optional
from pydantic import BaseModel, PlainSerializer, model_validator

# Serialize Decimal as string to preserve exact precision without fixed-point zero inflation
DecimalStr = Annotated[Decimal, PlainSerializer(lambda x: str(x), return_type=str)]

# Serialize date as ISO format string
DateStr = Annotated[date, PlainSerializer(lambda x: x.isoformat(), return_type=str)]


class Identifier(BaseModel):
    id: str
    url: Optional[str] = None
    value: Optional[str] = None


class Resource(BaseModel):
    id: str
    uri: str


class Concept(Resource):
    name: str
    nameFr: Optional[str] = None
    definition: Optional[str] = None
    isQuantity: Optional[bool] = None
    isRatio: Optional[bool] = None
    isRelationship: Optional[bool] = None
    exactMatch: Optional[list[str]] = None
    broader: Optional[list["Concept"]] = None
    narrower: Optional[list["Concept"]] = None
    parts: Optional[list["Concept"]] = None
    quantities: Optional[list["Quantity"]] = None
    quantity: Optional["Quantity"] = None


class Quantity(Resource):
    name: str
    definition: Optional[str] = None
    isRatio: Optional[bool] = None
    isRelationship: Optional[bool] = None
    constants: Optional[list["Constant"]] = None
    concepts: Optional[list[Concept]] = None


class Unit(Resource):
    name: Optional[str] = None
    ucumCode: Optional[str] = None
    siExpression: Optional[str] = None
    identifiers: Optional[list["Identifier"]] = None
    constants: Optional[list["Constant"]] = None

    @model_validator(mode="after")
    def set_default_name(self) -> "Unit":
        if self.name is None:
            self.name = self.id
        return self


class Version(Resource):
    published: Optional[DateStr] = None
    constants: Optional[list["Constant"]] = None


class ConstantValue(Resource):
    versionId: str
    value: str
    nistValue: Optional[str] = None
    valueSource: Optional[str] = None
    adjustmentRole: Optional[str] = None
    isComputed: Optional[bool] = None
    valueDecimal: Optional[DecimalStr] = None
    valueFloat: Optional[float] = None
    uncertainty: Optional[str] = None
    uncertaintyDecimal: Optional[DecimalStr] = None
    uncertaintyFloat: Optional[float] = None
    nistExponent: Optional[int] = None
    isExact: Optional[bool] = None
    isTruncated: Optional[bool] = None
    isNistTruncated: Optional[bool] = None
    significantDigits: Optional[int] = None
    fitsFloat32: Optional[bool] = None
    fitsFloat64: Optional[bool] = None
    recommendedNumericType: Optional[str] = None
    version: Optional[Version] = None
    identifiers: Optional[list[Identifier]] = None


class Constant(Resource):
    name: str
    nameFr: Optional[str] = None
    symbolTex: Optional[str] = None
    formula: Optional[str] = None
    formulaLatex: Optional[str] = None
    derivedFrom: Optional[list["Constant"]] = None
    usedIn: Optional[list["Constant"]] = None
    ucumCode: Optional[str] = None
    identifiers: Optional[list[Identifier]] = None
    isDeprecated: Optional[bool] = None
    quantity: Optional[Quantity] = None
    value: Optional[ConstantValue] = None
    unit: Optional[Unit] = None
    historicalValues: Optional[list[ConstantValue]] = None
