"""
High-Precision RDF Serializers for rdflib.

This module provides custom serializers for Turtle-based formats (Turtle, N3, TriG) 
that preserve the full precision of `xsd:double` and `xsd:decimal` literals.

By default, rdflib's `TurtleSerializer` truncates floating-point values to 5 decimal 
places in the shorthand notation (e.g., `6.62607015e-34` becomes `6.62607e-34`). 
These classes override the `label` method to use Python's standard string 
representation for doubles and full decimal expansion for decimals.

### Registration/Usage:

To use these serializers in an rdflib project:

```python
from rdflib.plugin import register
from rdflib.serializer import Serializer

# Register Turtle-HP
register(
    'turtle-hp',           # Format name for g.serialize(format='...')
    Serializer,
    'src.serializer',      # Module path
    'HighPrecisionTurtleSerializer'
)

# Usage
print(graph.serialize(format='turtle-hp'))
```
"""
from rdflib.plugins.serializers.turtle import TurtleSerializer
from rdflib.plugins.serializers.n3 import N3Serializer
from rdflib.plugins.serializers.trig import TrigSerializer
from rdflib.term import Literal
from rdflib.namespace import XSD

class HighPrecisionSerializerMixin:
    """
    A mixin that provides high-precision serialization for xsd:double and xsd:decimal.
    
    Overrides the default rdflib behavior of truncating floats to 5 decimal places.
    Ensures that values can be round-tripped without precision loss.
    """
    def label(self, node, position):
        if isinstance(node, Literal):
            if node.datatype == XSD.double:
                # Use Python's standard float-to-string conversion which is full precision
                val = float(node)
                res = str(val)
                # Ensure it remains a double in RDF shorthand (needs 'e' or '.')
                if "e" not in res.lower() and "." not in res:
                    res += ".0"
                return res
            elif node.datatype == XSD.decimal:
                # Ensure decimals don't use scientific notation and have at least one dot
                val = node.toPython()
                res = format(val, "f")
                if "." not in res:
                    res += ".0"
                return res
        
        return super().label(node, position)

class HighPrecisionTurtleSerializer(HighPrecisionSerializerMixin, TurtleSerializer):
    """
    High-precision Turtle serializer.
    Registered as 'turtle-hp'.
    """
    pass

class HighPrecisionN3Serializer(HighPrecisionSerializerMixin, N3Serializer):
    """
    High-precision N3 serializer.
    Registered as 'n3-hp'.
    """
    pass

class HighPrecisionTrigSerializer(HighPrecisionSerializerMixin, TrigSerializer):
    """
    High-precision TriG serializer.
    Registered as 'trig-hp'.
    """
    pass
