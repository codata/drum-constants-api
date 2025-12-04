from datetime import date
from decimal import Decimal
from typing import Annotated, Optional
from pydantic import BaseModel, Field, PlainSerializer, model_validator

# Serialize Decimal as string to preserve exact precision
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
    broader: Optional[list["Concept"]] = None
    parts: Optional[list["Concept"]] = None
    quantities: Optional[list["Quantity"]] = None
    quantity: Optional["Quantity"] = None

class Quantity(Resource):
    name: str
    constants: Optional[list["Constant"]] = None
    concepts: Optional[list[Concept]] = None

class Unit(Resource):
    name: Optional[str] = None
    identifiers: Optional[list["Identifier"]] = None
    constants: Optional[list["Constant"]] = None
    
    @model_validator(mode='after')
    def set_default_name(self) -> 'Unit':
        if self.name is None:
            self.name = self.id
        return self

class Version(Resource):
    published: Optional[DateStr] = None
    constants: Optional[list["Constant"]] = None

class ConstantValue(Resource):
    value: DecimalStr
    versionId: str
    uncertainty: Optional[DecimalStr] = None
    isExact: Optional[bool] = None
    isTruncated: Optional[bool] = None
    version: Optional[Version] = None
    identifiers: Optional[list[Identifier]] = None

class Constant(Resource):
    name: str
    quantity: Optional[Quantity] = None
    value: Optional[ConstantValue] = None
    unit: Optional[Unit] = None 
    historicalValues: Optional[list[ConstantValue]] = None

