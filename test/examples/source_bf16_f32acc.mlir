// The widened head structure, with independent small dimensions and no model identity.
module {
  func.func @forward(%a: tensor<2x3xbf16>, %w: tensor<3x5xbf16>) -> tensor<2x5xbf16> {
    %ae = tensor.empty() : tensor<2x3xf32>
    %aw = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a : tensor<2x3xbf16>) outs(%ae : tensor<2x3xf32>) {
    ^bb0(%x: bf16, %unused: f32):
      %value = arith.extf %x : bf16 to f32
      linalg.yield %value : f32
    } -> tensor<2x3xf32>
    %we = tensor.empty() : tensor<3x5xf32>
    %ww = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%w : tensor<3x5xbf16>) outs(%we : tensor<3x5xf32>) {
    ^bb0(%x: bf16, %unused: f32):
      %value = arith.extf %x : bf16 to f32
      linalg.yield %value : f32
    } -> tensor<3x5xf32>
    %zero = arith.constant 0.0 : f32
    %seed = tensor.splat %zero : tensor<2x5xf32>
    %mm = "linalg.matmul"(%aw, %ww, %seed) <{
      operandSegmentSizes = array<i32: 2, 1>,
      indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>,
                       affine_map<(d0,d1,d2)->(d2,d1)>,
                       affine_map<(d0,d1,d2)->(d0,d1)>]
    }> ({
    ^bb0(%left: f32, %right: f32, %accumulator: f32):
      %product = arith.mulf %left, %right : f32
      %sum = arith.addf %product, %accumulator : f32
      linalg.yield %sum : f32
    }) : (tensor<2x3xf32>, tensor<3x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %out = tensor.empty() : tensor<2x5xbf16>
    %y = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm : tensor<2x5xf32>) outs(%out : tensor<2x5xbf16>) {
    ^bb0(%x: f32, %unused: bf16):
      %value = arith.truncf %x : f32 to bf16
      linalg.yield %value : bf16
    } -> tensor<2x5xbf16>
    return %y : tensor<2x5xbf16>
  }
}
