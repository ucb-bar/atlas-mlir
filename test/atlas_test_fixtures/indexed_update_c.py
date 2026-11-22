"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""

def source_text(axis=1):
    shape = (2, 8, 4)
    update = list(shape)
    update[axis] = 1
    offsets = ['0'] * 3
    offsets[axis] = '-9223372036854775808'
    return f"""module {{\n  func.func @forward(%cache: tensor<2x8x4xf32>,\n      %update: tensor<{'x'.join(map(str, update))}xf32>,\n      %positions: tensor<1xi64>) -> tensor<2x8x4xf32> {{\n    %zero_index = arith.constant 0 : index\n    %position = tensor.extract %positions[%zero_index] : tensor<1xi64>\n    %zero = arith.constant 0 : i64\n    %limit = arith.constant {shape[axis]} : i64\n    %low = arith.cmpi sge, %position, %zero : i64\n    %high = arith.cmpi slt, %position, %limit : i64\n    %valid = arith.andi %low, %high : i1\n    cf.assert %valid, "index out of bounds"\n    %at = arith.index_cast %position : i64 to index\n    %result = "tensor.insert_slice"(%update,%cache,%at) <{{\n      static_offsets = array<i64: {','.join(offsets)}>,\n      static_sizes = array<i64: {','.join(map(str, update))}>,\n      static_strides = array<i64: 1,1,1>, operandSegmentSizes = array<i32: 1,1,1,0,0>\n    }}> : (tensor<{'x'.join(map(str, update))}xf32>, tensor<2x8x4xf32>, index) -> tensor<2x8x4xf32>\n    func.return %result : tensor<2x8x4xf32>\n  }}\n}}"""
