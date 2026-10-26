from torch.optim.lr_scheduler import LambdaLR


def modality_scaled_parameters(model, config):
    scales = config.get('modality_lr_scales', {})
    selected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        branch = ('shared' if 'BiShareAdapter' in name else
                  'vision' if '.visual.' in name else
                  'temperature' if name == 'temp' or name.endswith('.temp') else 'text')
        factor = float(scales.get(branch, 1.))
        if factor < 0:
            raise ValueError('Modality LR scales must be nonnegative')
        if factor != 1:
            selected.append((parameter, factor))
    return selected


def step_with_layer_scales(optimizer, selected):
    if not selected:
        optimizer.step()
        return
    # Temporary LR groups preserve the optimizer's exact decay semantics and
    # parameter-keyed moments. Restore canonical groups before schedule/save.
    scales = {id(p): factor for p, factor in selected}
    original = optimizer.param_groups
    temporary = []
    for group in original:
        buckets = {}
        for parameter in group['params']:
            buckets.setdefault(scales.get(id(parameter), 1.), []).append(parameter)
        for factor, parameters in buckets.items():
            temporary.append({**group, 'params': parameters, 'lr': group['lr'] * factor})
    try:
        optimizer.param_groups = temporary
        optimizer.step()
    finally:
        optimizer.param_groups = original


def create_scheduler(args, optimizer):
    if 'num_training_steps' not in args:
        args['num_training_steps'] = args['epochs'] * args['step_per_epoch']
    print("### num_training_steps, ", args['num_training_steps'], flush=True)

    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(args['num_training_steps'] * args['num_warmup_steps'])
    print("### num_warmup_steps, ", args['num_warmup_steps'], flush=True)

    if args.sched == 'linear':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    else:
        raise NotImplementedError(f"args.sched == {args.sched}")

    return lr_scheduler
