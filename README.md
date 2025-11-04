#### Please refer to : [tita_rl](https://github.com/DDTRobot/tita_rl) for environment setup

# Training and Policy-Exporting Guide for Wheeled Titaitit Task

## 1. Training

### 1.1 Start Training
You can start the training process with either of the following commands:

```bash
# Headless training (without visualization)
python train.py --task=wheeled_titatit --headless

# Or with visualization
python train.py --task=wheeled_titatit
```

## 2. Policy-Exporting

### 2.1 Uncomment the code
Uncomment the following code in modules/actor_critic.py around Line 293

```bash
def save_torch_jit_policy(self, path, device):
    print("ActorCriticBarlowTwins")
    obs_demo_input = torch.randn(1, self.num_prop).to(device)
    hist_demo_input = torch.randn(1, self.num_hist, self.num_prop).to(device)
    model_jit = torch.jit.trace(self.actor_teacher_backbone, (obs_demo_input, hist_demo_input))
    model_jit.save(path)
    torch_out = torch.onnx.export(self.actor_teacher_backbone,
                         (obs_demo_input, hist_demo_input),
                         "test.onnx",
                         opset_version=9
                         )
```
### 2.2 Excute Policy Testing
```bash
# without visualization
python simple_play.py --task=wheeled_titatit

# without visualization
python simple_play.py --task=wheeled_titatit --headless
```