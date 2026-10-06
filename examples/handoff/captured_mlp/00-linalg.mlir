builtin.module attributes {prov.weights_file = "../../../test/examples/captured_mlp32_weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x32xf32>, %1: tensor<32xf32>, %2: tensor<32x32xf32>, %3: tensor<32xf32>, %4: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %5 = tensor.empty() : tensor<32x32xf32>
    %6 = linalg.transpose ins(%0:tensor<32x32xf32>) outs(%5:tensor<32x32xf32>) permutation = [1, 0]
    %7 = tensor.empty() : tensor<32x32xf32>
    %8 = arith.constant {prov.module = "0"} 0.000000e+00 : f32
    %9 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "0"} ins(%8 : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %10 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "0", prov.fqn = "0"} ins(%4, %6 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%9 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %11 = tensor.empty() : tensor<32x32xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %1 : tensor<32x32xf32>, tensor<32xf32>) outs(%11 : tensor<32x32xf32>) attrs =  {prov.region_id = "add_0", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "0", prov.fqn = "0"} {
    ^bb0(%13: f32, %14: f32, %15: f32):
      %16 = arith.addf %13, %14 : f32
      linalg.yield %16 : f32
    } -> tensor<32x32xf32>
    %17 = tensor.empty() : tensor<32x32xf32>
    %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<32x32xf32>) outs(%17 : tensor<32x32xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "1", prov.fqn = "1"} {
    ^bb1(%19: f32, %20: f32):
      %21 = arith.constant 0.000000e+00 : f32
      %22 = arith.maximumf %19, %21 : f32
      linalg.yield %22 : f32
    } -> tensor<32x32xf32>
    %23 = tensor.empty() : tensor<32x32xf32>
    %24 = linalg.transpose ins(%2:tensor<32x32xf32>) outs(%23:tensor<32x32xf32>) permutation = [1, 0]
    %25 = tensor.empty() : tensor<32x32xf32>
    %26 = arith.constant {prov.module = "2"} 0.000000e+00 : f32
    %27 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "2"} ins(%26 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %28 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "2", prov.fqn = "2"} ins(%18, %24 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %29 = tensor.empty() : tensor<32x32xf32>
    %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%28, %3 : tensor<32x32xf32>, tensor<32xf32>) outs(%29 : tensor<32x32xf32>) attrs =  {prov.region_id = "add_1", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "2", prov.fqn = "2"} {
    ^bb2(%31: f32, %32: f32, %33: f32):
      %34 = arith.addf %31, %32 : f32
      linalg.yield %34 : f32
    } -> tensor<32x32xf32>
    func.return %30 : tensor<32x32xf32>
  }
}
