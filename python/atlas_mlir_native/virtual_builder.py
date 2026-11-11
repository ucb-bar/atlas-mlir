"""Small SSA printer shared by diagnostic consumers of Merlin schedules."""


class VirtualBuilder:
    def __init__(self, input_base, output_base):
        self.lines = ["module {", "  func.func @ordered_dense() -> !atlas.virtual_state attributes {"
                      f"atlas.input_dram_base = {input_base} : i64, atlas.output_dram_base = {output_base} : i64" + "} {",
                      '    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state']
        self.state, self.serial = "%s0", 0

    def pure(self, op, operands, types, result_type, attrs=""):
        self.serial += 1
        value = f"%v{self.serial}"
        self.lines.append(f'    {value} = "atlas.{op}"({", ".join(operands)}){attrs} : '
                          f'({", ".join(types)}) -> {result_type}')
        return value

    def emit(self, op, operands, types, result_type=None, attrs=""):
        self.serial += 1
        state, value = f"%s{self.serial}", f"%v{self.serial}"
        names = state if result_type is None else f"{state}, {value}"
        result = "!atlas.virtual_state" if result_type is None else f"(!atlas.virtual_state, {result_type})"
        self.lines.append(f'    {names} = "atlas.{op}"({", ".join([self.state, *operands])}){attrs} : '
                          f'({", ".join(["!atlas.virtual_state", *types])}) -> {result}')
        self.state = state
        return value

    def text(self):
        return "\n".join((*self.lines, f"    return {self.state} : !atlas.virtual_state", "  }", "}")) + "\n"
