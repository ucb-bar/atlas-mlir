module {
  func.func @two_mxu0_add_tile(
    %a: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
    %b: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
    %shared_weight: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"})
      -> tensor<32x32xbf16> {
    %zero = arith.constant 0.000000e+00 : bf16
    %empty_a = tensor.empty() : tensor<32x32xbf16>
    %init_a = linalg.fill ins(%zero : bf16) outs(%empty_a : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %first = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %shared_weight : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
      outs(%init_a : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "mxu0-e4m3-ordered-bf16-finite-v1",
               merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
               merlin.reduction_order = "k-ascending"} {
      ^bb0(%av: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
        %ab = arith.extf %av : f8E4M3FN to bf16
        %wb = arith.extf %wv : f8E4M3FN to bf16
        %next = math.fma %ab, %wb, %acc : bf16
        linalg.yield %next : bf16
    } -> tensor<32x32xbf16>
    %empty_b = tensor.empty() : tensor<32x32xbf16>
    %init_b = linalg.fill ins(%zero : bf16) outs(%empty_b : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %second = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%b, %shared_weight : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
      outs(%init_b : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "mxu0-e4m3-ordered-bf16-finite-v1",
               merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
               merlin.reduction_order = "k-ascending"} {
      ^bb0(%bv: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
        %bb = arith.extf %bv : f8E4M3FN to bf16
        %wb = arith.extf %wv : f8E4M3FN to bf16
        %next = math.fma %bb, %wb, %acc : bf16
        linalg.yield %next : bf16
    } -> tensor<32x32xbf16>
    %sum_storage = tensor.empty() : tensor<32x32xbf16>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%first, %second : tensor<32x32xbf16>, tensor<32x32xbf16>)
      outs(%sum_storage : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "selected-f32-rne-bf16-chop-v1",
               merlin.input_domain = "finite-normal-exact-fp32-sum-v1"} {
      ^bb0(%left: bf16, %right: bf16, %old: bf16):
        %lf = arith.extf %left : bf16 to f32
        %rf = arith.extf %right : bf16 to f32
        %fp32 = arith.addf %lf, %rf : f32
        %bits = arith.bitcast %fp32 : f32 to i32
        %shift = arith.constant 16 : i32
        %high = arith.shrui %bits, %shift : i32
        %narrow = arith.trunci %high : i32 to i16
        %bf16 = arith.bitcast %narrow : i16 to bf16
        linalg.yield %bf16 : bf16
    } -> tensor<32x32xbf16>
    func.return %sum : tensor<32x32xbf16>
  }
}
