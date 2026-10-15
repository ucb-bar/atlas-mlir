builtin.module attributes {prov.weights_file = "captured_attention_mlp32_weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x32xf32>, %1: tensor<32x32xf32>, %2: tensor<32x32xf32>, %3: tensor<32x32xf32>, %4: tensor<32x32xf32>, %5: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %6 = tensor.empty() : tensor<32x32xf32>
    %7 = linalg.transpose ins(%0:tensor<32x32xf32>) outs(%6:tensor<32x32xf32>) permutation = [1, 0]
    %8 = tensor.empty() : tensor<32x32xf32>
    %9 = arith.constant {prov.module = "query"} 0.000000e+00 : f32
    %10 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "query"} ins(%9 : f32) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %11 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "query", prov.fqn = "query", prov.transposed_b = "true"} ins(%5, %7 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%10 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %12 = tensor.empty() : tensor<32x32xf32>
    %13 = linalg.transpose ins(%1:tensor<32x32xf32>) outs(%12:tensor<32x32xf32>) permutation = [1, 0]
    %14 = tensor.empty() : tensor<32x32xf32>
    %15 = arith.constant {prov.module = "key"} 0.000000e+00 : f32
    %16 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "key"} ins(%15 : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %17 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "key", prov.fqn = "key", prov.transposed_b = "true"} ins(%5, %13 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %18 = tensor.empty() : tensor<32x32xf32>
    %19 = linalg.transpose ins(%2:tensor<32x32xf32>) outs(%18:tensor<32x32xf32>) permutation = [1, 0]
    %20 = tensor.empty() : tensor<32x32xf32>
    %21 = arith.constant {prov.module = "value"} 0.000000e+00 : f32
    %22 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "value"} ins(%21 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %23 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "value", prov.fqn = "value", prov.transposed_b = "true"} ins(%5, %19 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %24 = tensor.empty() : tensor<32x32xf32>
    %25 = linalg.transpose ins(%17:tensor<32x32xf32>) outs(%24:tensor<32x32xf32>) permutation = [1, 0]
    %26 = tensor.empty() : tensor<32x32xf32>
    %27 = arith.constant 0.000000e+00 : f32
    %28 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%27 : f32) outs(%26 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %29 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%11, %25 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %30 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %31 = tensor.splat %30 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<32x32xf32>
    %32 = tensor.empty() : tensor<32x32xf32>
    %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%29, %31 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%32 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%34: f32, %35: f32, %36: f32):
      %37 = arith.mulf %34, %35 : f32
      linalg.yield %37 : f32
    } -> tensor<32x32xf32>
    %38 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %39 = tensor.splat %38 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %40 = linalg.reduce ins(%33:tensor<32x32xf32>) outs(%39:tensor<32xf32>) dimensions = [1]
    (%41: f32, %42: f32) {
      %43 = arith.maximumf %41, %42 : f32
      linalg.yield %43 : f32
    }
    %44 = tensor.expand_shape %40 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %45 = tensor.empty() : tensor<32x32xf32>
    %46 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %44 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%45 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb1(%47: f32, %48: f32, %49: f32):
      %50 = arith.subf %47, %48 : f32
      linalg.yield %50 : f32
    } -> tensor<32x32xf32>
    %51 = tensor.empty() : tensor<32x32xf32>
    %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%46 : tensor<32x32xf32>) outs(%51 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb2(%53: f32, %54: f32):
      %55 = math.exp %53 : f32
      linalg.yield %55 : f32
    } -> tensor<32x32xf32>
    %56 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %57 = tensor.splat %56 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %58 = linalg.reduce ins(%52:tensor<32x32xf32>) outs(%57:tensor<32xf32>) dimensions = [1]
    (%59: f32, %60: f32) {
      %61 = arith.addf %59, %60 : f32
      linalg.yield %61 : f32
    }
    %62 = tensor.expand_shape %58 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %63 = tensor.empty() : tensor<32x32xf32>
    %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%52, %62 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%63 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb3(%65: f32, %66: f32, %67: f32):
      %68 = arith.divf %65, %66 : f32
      linalg.yield %68 : f32
    } -> tensor<32x32xf32>
    %69 = tensor.empty() : tensor<32x32xf32>
    %70 = arith.constant 0.000000e+00 : f32
    %71 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%70 : f32) outs(%69 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %72 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%64, %23 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%71 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %73 = tensor.empty() : tensor<32x32xf32>
    %74 = linalg.transpose ins(%3:tensor<32x32xf32>) outs(%73:tensor<32x32xf32>) permutation = [1, 0]
    %75 = tensor.empty() : tensor<32x32xf32>
    %76 = arith.constant {prov.module = "fc1"} 0.000000e+00 : f32
    %77 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "fc1"} ins(%76 : f32) outs(%75 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %78 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "fc1", prov.fqn = "fc1", prov.transposed_b = "true"} ins(%72, %74 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%77 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %79 = tensor.empty() : tensor<32x32xf32>
    %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%78 : tensor<32x32xf32>) outs(%79 : tensor<32x32xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32"} {
    ^bb4(%81: f32, %82: f32):
      %83 = arith.constant 0.000000e+00 : f32
      %84 = arith.maximumf %81, %83 : f32
      linalg.yield %84 : f32
    } -> tensor<32x32xf32>
    %85 = tensor.empty() : tensor<32x32xf32>
    %86 = linalg.transpose ins(%4:tensor<32x32xf32>) outs(%85:tensor<32x32xf32>) permutation = [1, 0]
    %87 = tensor.empty() : tensor<32x32xf32>
    %88 = arith.constant {prov.module = "fc2"} 0.000000e+00 : f32
    %89 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "fc2"} ins(%88 : f32) outs(%87 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %90 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "fc2", prov.fqn = "fc2", prov.transposed_b = "true"} ins(%80, %86 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%89 : tensor<32x32xf32>) -> tensor<32x32xf32>
    func.return %80, %90 : tensor<32x32xf32>, tensor<32x32xf32>
  }
}
