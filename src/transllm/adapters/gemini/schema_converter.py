"""Schema converter for JSON Schema to Gemini Schema transformation"""

from __future__ import annotations

from typing import Any, Dict


class GeminiSchemaConverter:
    """Convert JSON Schema to Gemini Schema format

    Implements a 6-stage transformation pipeline:
    1. Unpack $defs references
    2. Convert anyOf+null to nullable
    3. Convert type arrays to anyOf
    4. Fix enum restrictions
    5. Filter fields
    6. Add propertyOrdering for structured output
    """

    MAX_RECURSION_DEPTH = 50

    def __init__(self) -> None:
        self._recursion_depth = 0

    def convert(
        self, schema: Dict[str, Any], defs: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Convert JSON Schema to Gemini Schema

        Args:
            schema: The JSON Schema to convert
            defs: Optional $defs dictionary for reference resolution

        Returns:
            Gemini Schema format dictionary
        """
        if self._recursion_depth > self.MAX_RECURSION_DEPTH:
            raise ValueError(
                f"Max recursion depth ({self.MAX_RECURSION_DEPTH}) exceeded"
            )

        self._recursion_depth += 1

        try:
            # Stage 1: Unpack $defs references
            if defs:
                schema = self._unpack_defs(schema, defs)

            # Stage 2: Convert anyOf+null to nullable
            schema = self._convert_anyof_null(schema)

            # Stage 3: Convert type arrays to anyOf
            schema = self._convert_type_arrays(schema)

            # Stage 4: Fix enum restrictions
            schema = self._fix_enum(schema)

            # Stage 5: Filter fields
            schema = self._filter_fields(schema)

            # Stage 6: Add propertyOrdering for structured output
            if schema.get("type") == "object" and "properties" in schema:
                schema["propertyOrdering"] = list(schema["properties"].keys())

            return schema

        finally:
            self._recursion_depth -= 1

    def _unpack_defs(
        self, schema: Dict[str, Any], defs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively unpack $ref references from $defs"""
        if isinstance(schema, dict):
            # Handle $ref
            if "$ref" in schema:
                ref = schema["$ref"]
                if ref.startswith("#/$defs/"):
                    ref_name = ref.split("/")[-1]
                    if ref_name in defs:
                        return self._unpack_defs(defs[ref_name], defs)
                    else:
                        raise ValueError(f"Reference {ref_name} not found in defs")

            # Recursively process dict values
            return {k: self._unpack_defs(v, defs) for k, v in schema.items()}
        elif isinstance(schema, list):
            return [self._unpack_defs(item, defs) for item in schema]
        else:
            return schema

    def _convert_anyof_null(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert anyOf with null type to nullable field"""
        if isinstance(schema, dict):
            if "anyOf" in schema:
                anyof = schema["anyOf"]
                non_null_types = [
                    item
                    for item in anyof
                    if not (isinstance(item, dict) and item.get("type") == "null")
                ]

                if len(non_null_types) == 1 and non_null_types[0].get("type") != "null":
                    # Single non-null type + null → nullable
                    result = non_null_types[0].copy()
                    result["nullable"] = True
                    return result

            # Recursively process
            return {k: self._convert_anyof_null(v) for k, v in schema.items()}
        elif isinstance(schema, list):
            return [self._convert_anyof_null(item) for item in schema]
        else:
            return schema

    def _convert_type_arrays(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert type arrays to anyOf format"""
        if isinstance(schema, dict):
            if "type" in schema and isinstance(schema["type"], list):
                # Convert type array to anyOf
                anyof = [{"type": t} for t in schema["type"]]
                result = {"anyOf": anyof}
                # Copy other fields except 'type'
                for k, v in schema.items():
                    if k != "type":
                        result[k] = v
                return result

            # Recursively process
            return {k: self._convert_type_arrays(v) for k, v in schema.items()}
        elif isinstance(schema, list):
            return [self._convert_type_arrays(item) for item in schema]
        else:
            return schema

    def _fix_enum(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Fix enum restrictions - only allow for string types"""
        if isinstance(schema, dict):
            if "enum" in schema:
                # Only keep enum if type is string or not specified
                if schema.get("type") != "string":
                    # Remove enum for non-string types
                    schema = schema.copy()
                    schema.pop("enum", None)

            # Recursively process
            return {k: self._fix_enum(v) for k, v in schema.items()}
        elif isinstance(schema, list):
            return [self._fix_enum(item) for item in schema]
        else:
            return schema

    def _filter_fields(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to only Gemini Schema valid fields"""
        if not isinstance(schema, dict):
            return schema

        # Gemini Schema valid fields
        valid_fields = {
            "type",
            "properties",
            "items",
            "required",
            "nullable",
            "enum",
            "propertyOrdering",
            "description",
            "anyOf",
            "allOf",
        }

        # Filter fields
        filtered = {k: v for k, v in schema.items() if k in valid_fields}

        # Recursively process nested schemas
        for key, value in filtered.items():
            if isinstance(value, dict):
                filtered[key] = self._filter_fields(value)
            elif isinstance(value, list):
                filtered[key] = [
                    self._filter_fields(item) if isinstance(item, dict) else item
                    for item in value
                ]

        return filtered

    def validate_schema(self, schema: Dict[str, Any]) -> None:
        """Validate Gemini Schema for edge cases"""
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                if "properties" in schema:
                    if not schema["properties"]:
                        raise ValueError("Object schema cannot have empty properties")

                if "items" in schema:
                    if not schema["items"]:
                        schema["items"] = {"type": "object"}

            # Check anyOf with only null
            if "anyOf" in schema:
                if (
                    len(schema["anyOf"]) == 1
                    and schema["anyOf"][0].get("type") == "null"
                ):
                    raise ValueError("anyOf cannot contain only null type")

            # Recursively validate
            for value in schema.values():
                if isinstance(value, (dict, list)):
                    self.validate_schema(value)
        elif isinstance(schema, list):
            for item in schema:
                if isinstance(item, (dict, list)):
                    self.validate_schema(item)


def convert_json_schema_to_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to convert JSON Schema to Gemini Schema

    Args:
        schema: JSON Schema dictionary

    Returns:
        Gemini Schema format dictionary
    """
    converter = GeminiSchemaConverter()

    # Extract $defs if present
    defs = schema.pop("$defs", None) if isinstance(schema, dict) else None

    # Convert schema
    result = converter.convert(schema, defs)

    # Validate result
    converter.validate_schema(result)

    return result
