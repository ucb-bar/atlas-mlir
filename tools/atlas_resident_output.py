"""Common BF16 pair retention in spare VMEM banks 2 and 3."""

RESIDENT_H_WORD = 524288 // 4


def retain_output(epilogue, output_reg):
    epilogue._check_output_register(output_reg)
    lines = epilogue._delta(output_reg) + ['srli @s0, @s0, 2', f'li @s1, {RESIDENT_H_WORD}',
             'add @s0, @s0, @s1', 'vstore m2, 0(@s0)', 'delay 34', 'addi @s0, @s0, 256',
             'vstore m3, 0(@s0)', 'delay 34']
    return epilogue._registers(lines)
