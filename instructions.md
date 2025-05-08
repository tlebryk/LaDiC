## Setting up shell in modal:

```
 pip install modal
  modal setup
```

## Debugging using a shell

Open a shell
```bash
modal shell --image  tlebryk/ladic:latest --gpu t4
```

Run inference on an image quickly: 
```bash
python infer.py
```

Run training: (see readme)
```bash
accelerate launch main.py [--args]
```

## Running directly over modal



