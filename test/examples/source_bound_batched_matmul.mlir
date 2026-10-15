builtin.module attributes {prov.weights_file = "weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<2x2x2xf32>, %b: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<2x2x2xf32>
    %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(b,m,n,k)->(b,m,k)>,
                       affine_map<(b,m,n,k)->(b,k,n)>,
                       affine_map<(b,m,n,k)->(b,m,n)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2x2x2xf32>, tensor<2x2x2xf32>)
      outs(%init : tensor<2x2x2xf32>) {
      ^bb0(%av: f32, %bv: f32, %acc: f32):
        %product = arith.mulf %av, %bv : f32
        %sum = arith.addf %acc, %product : f32
        linalg.yield %sum : f32
    } -> tensor<2x2x2xf32>
    func.return %result : tensor<2x2x2xf32>
  }
}
