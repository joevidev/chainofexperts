```
base command:
python main.py 
```

```
try to use a different layer&iter setting:
python main.py --config config/base.yaml  model.output_dir=".output/coe-l1t1"  model.config.num_hidden_layers=1  model.config.inner_iter=1
```

```
try to use a different dataset:
CUDA_VISIBLE_DEVICES=0 python main.py --config config/math.yaml  model.output_dir=".output/coe-math-l1t1" model.config.num_hidden_layers=1 model.config.inner_iter=1
CUDA_VISIBLE_DEVICES=1 python main.py --config config/math.yaml  model.output_dir=".output/coe-math-l1t2" model.config.num_hidden_layers=1 model.config.inner_iter=2
CUDA_VISIBLE_DEVICES=2 python main.py --config config/math.yaml  model.output_dir=".output/coe-math-l2t1" model.config.num_hidden_layers=2 model.config.inner_iter=1
CUDA_VISIBLE_DEVICES=3 python main.py --config config/math.yaml  model.output_dir=".output/coe-math-l3t1" model.config.num_hidden_layers=3 model.config.inner_iter=1
```

python main.py --config config/math.yaml  model.output_dir=".output/coe-math-l1e256t4" model.config.num_hidden_layers=1 model.config.n_routed_experts=256 model.config.inner_iter=4 && python main.py --config config/math.yaml  model.output_dir=".output/coe-math-l4e64t1" model.config.num_hidden_layers=4 model.config.n_routed_experts=64 model.config.inner_iter=1 && sleep 5 && python main.py --config config/math.yaml  model.output_dir=".output/coe-math-l2e128t2" model.config.num_hidden_layers=2 model.config.n_routed_experts=128 model.config.inner_iter=2 && sleep 5
