"""Named i32 SSA construction shared by device scalar lowerings."""
class ScalarBuilder:
    def __init__(self, lines):
        self.lines = lines
        self.serial = 0

    def value(self, expression, type_='i32'):
        name = f'%scalar{self.serial}'
        self.serial += 1
        self.lines.append(f'    {name} = {expression}')
        return name

    def constant(self, number):
        number &= 0xffffffff
        return self.value(f'arith.constant {number if number < 2**31 else number - 2**32} : i32')

    def binary(self, kind, left, right):
        operation = {'add':'addi', 'sub':'subi', 'and':'andi', 'or':'ori',
                     'xor':'xori', 'shl':'shli', 'shr':'shrui'}[kind]
        return self.value(f'arith.{operation} {left}, {right} : i32')

    def compare(self, predicate, left, right):
        test = self.value(f'arith.cmpi {predicate}, {left}, {right} : i32')
        return self.value(f'arith.extui {test} : i1 to i32')

    def select(self, yes, no, predicate):
        mask = self.binary('sub', self.constant(0), predicate)
        return self.binary('xor', no, self.binary('and', self.binary('xor', yes, no), mask))
