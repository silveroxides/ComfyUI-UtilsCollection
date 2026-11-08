import math
from typing import Any, Callable

from comfy_api.latest import io


MATH_CATEGORY = "utils/math"
LOGIC_CATEGORY = "utils/logic"


def _math_template():
    return io.MatchType.Template("math", allowed_types=[io.Int, io.Float])


def _math_autogrow(id: str, prefix: str):
    return io.Autogrow.Input(id, template=io.Autogrow.TemplatePrefix(io.MatchType.Input(prefix, template=_math_template()), prefix=prefix, min=2, max=10))


def _boolean_autogrow():
    return io.Autogrow.Input("inputs", template=io.Autogrow.TemplatePrefix(io.Boolean.Input("input"), prefix="input", min=2, max=10))


class UC_LogicIF(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="UC_LogicIF", display_name="IF", category=LOGIC_CATEGORY, inputs=[io.Boolean.Input("if_condition"), io.AnyType.Input("when_true"), io.AnyType.Input("when_false", optional=True)], outputs=[io.AnyType.Output(display_name="result")])

    @classmethod
    def execute(cls, if_condition: bool, when_true: Any, when_false: Any = None):
        return io.NodeOutput(when_true if if_condition else when_false)


class _BooleanAutogrowNode(io.ComfyNode):
    node_id = ""
    display_name = ""
    operation: Callable[[list[bool]], bool]

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.node_id, display_name=cls.display_name, category=LOGIC_CATEGORY, inputs=[_boolean_autogrow()], outputs=[io.Boolean.Output()])

    @classmethod
    def execute(cls, inputs: io.Autogrow.Type):
        return io.NodeOutput(cls.operation(list(inputs.values())))


class UC_LogicAND(_BooleanAutogrowNode):
    node_id = "UC_LogicAND"
    display_name = "AND"
    operation = staticmethod(all)


class UC_LogicOR(_BooleanAutogrowNode):
    node_id = "UC_LogicOR"
    display_name = "OR"
    operation = staticmethod(any)


class UC_LogicXOR(_BooleanAutogrowNode):
    node_id = "UC_LogicXOR"
    display_name = "XOR"
    operation = staticmethod(lambda values: sum(bool(value) for value in values) % 2 == 1)


class UC_LogicNOT(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="UC_LogicNOT", display_name="NOT", category=LOGIC_CATEGORY, inputs=[io.Boolean.Input("input")], outputs=[io.Boolean.Output()])

    @classmethod
    def execute(cls, input: bool):
        return io.NodeOutput(not input)


class _VariadicMathNode(io.ComfyNode):
    node_id = ""
    display_name = ""
    operation: Callable[[Any, Any], Any]

    @classmethod
    def define_schema(cls):
        template = _math_template()
        return io.Schema(node_id=cls.node_id, display_name=cls.display_name, category=MATH_CATEGORY, inputs=[_math_autogrow("operands", "operand")], outputs=[io.MatchType.Output(template=template)])

    @classmethod
    def execute(cls, operands: io.Autogrow.Type):
        values = iter(operands.values())
        result = next(values)
        for value in values:
            result = cls.operation(result, value)
        return io.NodeOutput(result)


class UC_MathAdd(_VariadicMathNode):
    node_id = "UC_MathAdd"
    display_name = "Add"
    operation = staticmethod(lambda left, right: left + right)


class UC_MathSubtract(_VariadicMathNode):
    node_id = "UC_MathSubtract"
    display_name = "Subtract"
    operation = staticmethod(lambda left, right: left - right)


class UC_MathMultiply(_VariadicMathNode):
    node_id = "UC_MathMultiply"
    display_name = "Multiply"
    operation = staticmethod(lambda left, right: left * right)


class UC_MathDivide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _math_template()
        return io.Schema(node_id="UC_MathDivide", display_name="Divide", category=MATH_CATEGORY, inputs=[_math_autogrow("operands", "operand"), io.Boolean.Input("handle_zero", default=True)], outputs=[io.MatchType.Output(template=template)])

    @classmethod
    def validate_inputs(cls, operands: io.Autogrow.Type, handle_zero: bool):
        if not handle_zero and any(value == 0 for value in list(operands.values())[1:]):
            return "Division by zero is not allowed"
        return True

    @classmethod
    def execute(cls, operands: io.Autogrow.Type, handle_zero: bool):
        values = iter(operands.values())
        result = next(values)
        for value in values:
            if value == 0:
                if handle_zero:
                    return io.NodeOutput(0)
                raise ValueError("Division by zero is not allowed")
            result /= value
        return io.NodeOutput(result)


class UC_MathPower(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _math_template()
        return io.Schema(node_id="UC_MathPower", display_name="Power", category=MATH_CATEGORY, inputs=[io.MatchType.Input("base", template=template), io.MatchType.Input("exponent", template=template)], outputs=[io.MatchType.Output(template=template)])

    @classmethod
    def execute(cls, base: Any, exponent: Any):
        return io.NodeOutput(base**exponent)


class _UnaryIntegerNode(io.ComfyNode):
    node_id = ""
    display_name = ""
    operation: Callable[[Any], int]

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.node_id, display_name=cls.display_name, category=MATH_CATEGORY, inputs=[io.MatchType.Input("value", template=_math_template())], outputs=[io.Int.Output(display_name="result")])

    @classmethod
    def execute(cls, value: Any):
        return io.NodeOutput(cls.operation(value))


class UC_MathFloor(_UnaryIntegerNode):
    node_id = "UC_MathFloor"
    display_name = "Floor"
    operation = staticmethod(math.floor)


class UC_MathCeil(_UnaryIntegerNode):
    node_id = "UC_MathCeil"
    display_name = "Ceil"
    operation = staticmethod(math.ceil)


class UC_MathRound(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _math_template()
        return io.Schema(node_id="UC_MathRound", display_name="Round", category=MATH_CATEGORY, inputs=[io.MatchType.Input("value", template=template), io.Int.Input("decimals", default=0, min=0, max=10)], outputs=[io.MatchType.Output(template=template)])

    @classmethod
    def execute(cls, value: Any, decimals: int):
        return io.NodeOutput(round(value, decimals))


class UC_MathModulo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _math_template()
        return io.Schema(node_id="UC_MathModulo", display_name="Modulo", category=MATH_CATEGORY, inputs=[io.MatchType.Input("value_a", template=template), io.MatchType.Input("value_b", template=template)], outputs=[io.MatchType.Output(template=template)])

    @classmethod
    def execute(cls, value_a: Any, value_b: Any):
        return io.NodeOutput(0 if value_b == 0 else value_a % value_b)


class UC_MathAbs(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _math_template()
        return io.Schema(node_id="UC_MathAbs", display_name="Absolute", category=MATH_CATEGORY, inputs=[io.MatchType.Input("value", template=template)], outputs=[io.MatchType.Output(template=template)])

    @classmethod
    def execute(cls, value: Any):
        return io.NodeOutput(abs(value))


class UC_MathSqrt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="UC_MathSqrt", display_name="Square Root", category=MATH_CATEGORY, inputs=[io.MatchType.Input("value", template=_math_template())], outputs=[io.Float.Output(display_name="result")])

    @classmethod
    def execute(cls, value: Any):
        return io.NodeOutput(0.0 if value < 0 else math.sqrt(value))


class _TrigonometryNode(io.ComfyNode):
    node_id = ""
    display_name = ""
    operation: Callable[[float], float]

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.node_id, display_name=cls.display_name, category=f"{MATH_CATEGORY}/trigonometry", inputs=[io.MatchType.Input("angle", template=_math_template()), io.Combo.Input("unit", options=["Radians", "Degrees"], default="Radians")], outputs=[io.Float.Output(display_name="result")])

    @classmethod
    def execute(cls, angle: Any, unit: str):
        if unit == "Degrees":
            angle = math.radians(angle)
        return io.NodeOutput(cls.operation(angle))


class UC_MathSin(_TrigonometryNode):
    node_id = "UC_MathSin"
    display_name = "Sine"
    operation = staticmethod(math.sin)


class UC_MathCos(_TrigonometryNode):
    node_id = "UC_MathCos"
    display_name = "Cosine"
    operation = staticmethod(math.cos)


class UC_MathTan(_TrigonometryNode):
    node_id = "UC_MathTan"
    display_name = "Tangent"
    operation = staticmethod(math.tan)


class _MinMaxNode(io.ComfyNode):
    node_id = ""
    display_name = ""
    operation: Callable

    @classmethod
    def define_schema(cls):
        template = _math_template()
        return io.Schema(node_id=cls.node_id, display_name=cls.display_name, category=MATH_CATEGORY, inputs=[_math_autogrow("values", "value")], outputs=[io.MatchType.Output(template=template)])

    @classmethod
    def execute(cls, values: io.Autogrow.Type):
        return io.NodeOutput(cls.operation(values.values()))


class UC_MathMin(_MinMaxNode):
    node_id = "UC_MathMin"
    display_name = "Minimum"
    operation = staticmethod(min)


class UC_MathMax(_MinMaxNode):
    node_id = "UC_MathMax"
    display_name = "Maximum"
    operation = staticmethod(max)


class UC_MathClamp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _math_template()
        return io.Schema(node_id="UC_MathClamp", display_name="Clamp", category=MATH_CATEGORY, inputs=[io.MatchType.Input("value", template=template), io.MatchType.Input("min_value", template=template), io.MatchType.Input("max_value", template=template)], outputs=[io.MatchType.Output(template=template)])

    @classmethod
    def execute(cls, value: Any, min_value: Any, max_value: Any):
        return io.NodeOutput(max(min(value, max_value), min_value))


class UC_MathNumberConvert(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="UC_MathNumberConvert", display_name="Number Convert", category=MATH_CATEGORY, inputs=[io.MatchType.Input("number_value", template=_math_template())], outputs=[io.Int.Output("result_int", display_name="result_int"), io.Float.Output("result_float", display_name="result_float")])

    @classmethod
    def execute(cls, number_value: Any):
        return io.NodeOutput(int(number_value), float(number_value))


class UC_StringToNumber(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _math_template()
        return io.Schema(node_id="UC_StringToNumber", display_name="String To Number", category=MATH_CATEGORY, inputs=[io.String.Input("string", multiline=False), io.MatchType.Input("default_value", template=template, optional=True)], outputs=[io.MatchType.Output(template=template, display_name="result")])

    @classmethod
    def execute(cls, string: str, default_value: Any = 0):
        try:
            return io.NodeOutput(float(string) if "." in string else int(string))
        except (ValueError, TypeError):
            return io.NodeOutput(default_value)


class UC_NumberToString(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="UC_NumberToString", display_name="Number To String", category=MATH_CATEGORY, inputs=[io.MatchType.Input("number", template=_math_template())], outputs=[io.String.Output(display_name="result")])

    @classmethod
    def execute(cls, number: Any):
        return io.NodeOutput(str(number))


class UC_MathCompare(io.ComfyNode):
    OPERATIONS = {
        "Equal": lambda left, right: left == right,
        "Not Equal": lambda left, right: left != right,
        "Greater Than": lambda left, right: left > right,
        "Less Than": lambda left, right: left < right,
        "Greater Than or Equal": lambda left, right: left >= right,
        "Less Than or Equal": lambda left, right: left <= right,
    }

    @classmethod
    def define_schema(cls):
        template = _math_template()
        return io.Schema(
            node_id="UC_MathCompare",
            display_name="Compare",
            category=MATH_CATEGORY,
            inputs=[io.MatchType.Input("value_a", template=template), io.MatchType.Input("value_b", template=template), io.Combo.Input("comparison", options=list(cls.OPERATIONS), default="Equal")],
            outputs=[io.Boolean.Output(display_name="result")],
        )

    @classmethod
    def execute(cls, value_a: Any, value_b: Any, comparison: str):
        operation = cls.OPERATIONS.get(comparison)
        return io.NodeOutput(False if operation is None else operation(value_a, value_b))


class UC_MathOperation(io.ComfyNode):
    OPERATIONS = {"Add": lambda left, right: left + right, "Subtract": lambda left, right: left - right, "Multiply": lambda left, right: left * right, "Divide": lambda left, right: left / right}

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="UC_MathOperation", display_name="Math Operation (Example)", category=MATH_CATEGORY, inputs=[io.AnyType.Input("value_a"), io.AnyType.Input("value_b"), io.Combo.Input("operation", options=list(cls.OPERATIONS), default="Add")], outputs=[io.AnyType.Output(display_name="result")])

    @classmethod
    def execute(cls, value_a: Any, value_b: Any, operation: str):
        function = cls.OPERATIONS.get(operation)
        return io.NodeOutput(0 if function is None else function(value_a, value_b))


class UC_MathAspectRatio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MathAspectRatio", display_name="Aspect Ratio", category=MATH_CATEGORY, inputs=[io.Int.Input("width", default=1920, min=1), io.Int.Input("height", default=1080, min=1)], outputs=[io.Int.Output("ratio_width", display_name="ratio_width"), io.Int.Output("ratio_height", display_name="ratio_height")]
        )

    @classmethod
    def execute(cls, width: int, height: int):
        divisor = math.gcd(width, height)
        return io.NodeOutput(width // divisor, height // divisor)


LOGIC_MATH_NODES = [
    UC_MathAdd,
    UC_MathSubtract,
    UC_MathMultiply,
    UC_MathDivide,
    UC_MathPower,
    UC_MathFloor,
    UC_MathCeil,
    UC_MathRound,
    UC_MathModulo,
    UC_MathAbs,
    UC_MathSqrt,
    UC_MathSin,
    UC_MathCos,
    UC_MathTan,
    UC_MathMin,
    UC_MathMax,
    UC_MathClamp,
    UC_MathNumberConvert,
    UC_StringToNumber,
    UC_NumberToString,
    UC_MathCompare,
    UC_MathOperation,
    UC_MathAspectRatio,
    UC_LogicIF,
    UC_LogicAND,
    UC_LogicOR,
    UC_LogicNOT,
    UC_LogicXOR,
]
