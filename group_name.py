"""Adapted MobileNetV2 reproduction for CSCI935. Run with --help."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

CLASSES = ['Brown Spot', 'Leaf Scald', 'Rice Blast', 'Rice Tungro', 'Sheath Blight']
ALIASES = [['brownspot', 'browonspot'], ['leafscald', 'leafscaled'], ['riceblast', 'leafblast'],
           ['ricetungro', 'riceturgro'], ['sheathblight', 'shathblight']]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
WORK = '_mobilenet'


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def token(text):
    return re.sub('[^a-z]', '', text.lower())


def scan(root):
    """Infer labels from ancestor folders only; never from image filenames."""
    if not root.is_dir():
        raise ValueError(f'Dataset folder does not exist: {root}')
    rows, seen = [], {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if WORK in relative.parts or path.suffix.lower() not in EXTENSIONS:
            continue
        folders = [token(p) for p in relative.parts[:-1]]
        labels = [i for i, aliases in enumerate(ALIASES) if any(a in folders for a in aliases)]
        backgrounds = set()
        for folder in folders:
            if 'white' in folder:
                backgrounds.add('white')
            if 'field' in folder:
                backgrounds.add('field')
        if len(labels) != 1 or len(backgrounds) != 1:
            raise ValueError(f'Cannot infer one class/background for {relative}. Use a manifest (see README).')
        with Image.open(path) as im:
            im.verify()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = {'path': relative.as_posix(), 'label': labels[0], 'background': backgrounds.pop(), 'sha256': digest}
        if digest in seen and (seen[digest]['label'], seen[digest]['background']) != (row['label'], row['background']):
            raise ValueError(f'Identical files have conflicting class/background labels: {relative}')
        seen[digest] = row
        rows.append(row)
    if not rows:
        raise ValueError('No labelled images found.')
    return rows


def load_rows(root, manifest=None):
    if not manifest:
        return scan(root)
    rows, seen, paths = [], {}, set()
    with Path(manifest).open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            label = CLASSES.index(r['class'])
            bg = r['background'].lower()
            path = (root / r['path']).resolve()
            if bg not in ('field', 'white') or not path.is_relative_to(root.resolve()):
                raise ValueError('Manifest backgrounds must be field/white and paths must stay inside dataset.')
            if path in paths or WORK in path.relative_to(root.resolve()).parts:
                raise ValueError('Repeated manifest path or generated output included.')
            paths.add(path)
            with Image.open(path) as im:
                im.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != (label, bg):
                raise ValueError('Identical files have conflicting manifest labels.')
            seen[digest] = (label, bg)
            rows.append({'path': path.relative_to(root.resolve()).as_posix(), 'label': label,
                         'background': bg, 'sha256': digest})
    if not rows:
        raise ValueError('Empty manifest.')
    return rows


def splits(rows, seed):
    # Split unique hashes, then expand: exact duplicates cannot cross a boundary.
    unique = list({r['sha256']: r for r in rows}.values())
    strata = lambda rs: [f"{r['label']}_{r['background']}" for r in rs]
    if set(strata(unique)) != {f'{i}_{b}' for i in range(5) for b in ('field', 'white')}:
        raise ValueError('All five classes must occur in both backgrounds.')
    trainval, test = train_test_split(unique, test_size=.20, random_state=seed, stratify=strata(unique))
    train, val = train_test_split(trainval, test_size=.25, random_state=seed, stratify=strata(trainval))
    allocation = {r['sha256']: name for name, subset in [('train', train), ('validation', val), ('test', test)] for r in subset}
    return [{**r, 'split': allocation[r['sha256']]} for r in rows]


def audit(args):
    root = Path(args.data).resolve()
    rows = load_rows(root, args.manifest)
    counts = Counter((CLASSES[r['label']], r['background']) for r in rows)
    result = {'images': len(rows), 'unique_sha256': len(set(r['sha256'] for r in rows)),
              'counts': [{'class': c, 'background': b, 'n': counts[c, b]} for c in CLASSES for b in ('field', 'white')]}
    dump(root / WORK / 'audit.json', result)
    write_csv(root / WORK / 'inventory.csv', rows)
    for seed in (42, 43, 44, 45, 46):
        splits(rows, seed)
    print(json.dumps(result, indent=2))
    print('All five planned stratified splits are feasible. Inspect images for near duplicates before training.')


def torch_modules():
    import torch
    import torchvision
    from torchvision import transforms
    return torch, torchvision, transforms


def weights_path(root):
    return root / WORK / 'weights' / 'mobilenet_v2-b0353104.pth'


def download(args):
    torch, vision, _ = torch_modules()
    path = weights_path(Path(args.data).resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(vision.models.MobileNet_V2_Weights.IMAGENET1K_V1.url,
                                 str(path), hash_prefix='b0353104')
    print(path)


def make_model(head, dropout, weight_file=None):
    torch, vision, _ = torch_modules()
    base = vision.models.mobilenet_v2(weights=None)
    if weight_file is not None:
        base.load_state_dict(torch.load(weight_file, map_location='cpu', weights_only=True))

    class FrozenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = base.features
            for p in self.features.parameters():
                p.requires_grad_(False)
            first = torch.nn.Flatten() if head == 'paper' else torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
            size = 62720 if head == 'paper' else 1280
            self.head = torch.nn.Sequential(first, torch.nn.Linear(size, 64), torch.nn.ReLU(),
                                           torch.nn.Dropout(dropout), torch.nn.Linear(64, 5))
        def train(self, mode=True):
            super().train(mode)
            # Freeze BatchNorm running statistics as well as its parameters.
            self.features.eval()
            return self
        def forward(self, x):
            with torch.no_grad():
                features = self.features(x)
            return self.head(features)
    return FrozenModel()


def transforms_for(training, augment):
    _, _, T = torch_modules()
    ops = [T.Resize((224, 224))]
    if training and augment:
        ops += [T.RandomAffine(degrees=30, scale=(.8, 1.2)), T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(), T.ColorJitter(brightness=(.8, 1.2))]
    return T.Compose(ops + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


def loader(root, rows, batch_size, training=False, augment=True):
    torch, _, _ = torch_modules()
    transform = transforms_for(training, augment)
    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            r = rows[index]
            with Image.open(root / r['path']) as im:
                x = transform(ImageOps.exif_transpose(im).convert('RGB'))
            return x, r['label']
    # num_workers=0 avoids Windows spawn issues and keeps memory use predictable.
    return torch.utils.data.DataLoader(Images(), batch_size=batch_size, shuffle=training, num_workers=0)


def epoch(model, batches, optimizer=None, class_weights=None):
    torch, _, _ = torch_modules()
    model.train(optimizer is not None)
    total, loss_sum, loss_denominator, correct, predictions, truths = 0, 0., 0., 0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in batches:
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, weight=class_weights)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            pred = logits.argmax(1)
            total += len(y)
            denominator = len(y) if class_weights is None else class_weights[y].sum().item()
            loss_sum += loss.item() * denominator
            loss_denominator += denominator
            correct += (pred == y).sum().item()
            predictions.extend(pred.tolist())
            truths.extend(y.tolist())
    return loss_sum / loss_denominator, correct / total, truths, predictions


def metrics(y, pred):
    labels = list(range(5))
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
            'classification_report': classification_report(y, pred, labels=labels, target_names=CLASSES,
                                                           output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=labels).tolist()}


def plot_outputs(folder, history, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for prefix in ('train', 'val'):
        ax[0].plot([r['epoch'] for r in history], [r[prefix + '_loss'] for r in history], label=prefix)
        ax[1].plot([r['epoch'] for r in history], [r[prefix + '_accuracy'] for r in history], label=prefix)
    for a, title in zip(ax, ('Cross entropy', 'Accuracy')):
        a.set_title(title); a.set_xlabel('Epoch'); a.legend()
    fig.tight_layout(); fig.savefig(folder / 'learning_curves.png'); plt.close(fig)
    for bg, result in results.items():
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = np.array(result['confusion_matrix'])
        ax.imshow(cm, cmap='Blues')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
        ax.set_xticks(range(5), CLASSES, rotation=35, ha='right'); ax.set_yticks(range(5), CLASSES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{bg} test set (counts)')
        fig.tight_layout(); fig.savefig(folder / f'confusion_{bg}.png'); plt.close(fig)


def train(args):
    torch, _, _ = torch_modules()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    root = Path(args.data).resolve()
    rows = load_rows(root, args.manifest)
    path = weights_path(root)
    if not path.is_file():
        raise ValueError('Pretrained weights missing. Run download-weights first.')
    # Fail before creating outputs if any split is infeasible or would overwrite work.
    all_splits = {seed: splits(rows, seed) for seed in args.seeds}
    experiment = root / WORK / 'runs' / args.run
    if experiment.exists():
        raise ValueError(f'Run already exists: {experiment}. Choose a new --run name.')
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn', 'matplotlib')}
    config = {k: v for k, v in vars(args).items() if k != 'func'}
    config.update(classes=CLASSES, split_proportions=[.6, .2, .2], versions=versions,
                  checkpoint_rules={'best.pt': 'minimum unweighted validation cross entropy',
                                    'best_macro_f1.pt': 'maximum mixed validation macro F1; earliest epoch wins ties'},
                  stopping_rule='patience on minimum unweighted validation cross entropy',
                  weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), python=sys.version,
                  dataset_inventory_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
    dump(experiment / 'config.json', config)
    for seed, allocated in all_splits.items():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        folder = experiment / f'seed_{seed}'
        write_csv(folder / 'splits.csv', allocated)
        subsets = {s: [r for r in allocated if r['split'] == s] for s in ('train', 'validation', 'test')}
        batches = {s: loader(root, rs, args.batch_size, s == 'train', not args.no_augment) for s, rs in subsets.items()}
        model = make_model(args.head, args.dropout, path)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)
        counts = np.bincount([r['label'] for r in subsets['train']], minlength=5)
        if np.any(counts == 0):
            raise ValueError('All five classes must be present in training.')
        class_weights = (torch.tensor(len(subsets['train']) / (5 * counts), dtype=torch.float32)
                         if args.class_weighted else None)
        dump(folder / 'training_classes.json', {'classes': CLASSES, 'counts': counts.tolist(),
             'weights': class_weights.tolist() if class_weights is not None else None,
             'formula': 'N_train / (5 * n_class)' if args.class_weighted else 'unweighted'})
        best, stale, history, started = float('inf'), 0, [], time.perf_counter()
        best_epoch = 0
        best_f1, best_f1_epoch = float('-inf'), 0
        for n in range(1, args.epochs + 1):
            begin = time.perf_counter()
            tl, ta, _, _ = epoch(model, batches['train'], optimizer, class_weights)
            vl, va, vy, vp = epoch(model, batches['validation'])
            validation = scenario_metrics(subsets['validation'], vy, vp)
            history.append({'epoch': n, 'train_loss': tl, 'train_accuracy': ta, 'val_loss': vl,
                            'val_accuracy': va, 'val_macro_f1': validation['mixed']['macro_f1'],
                            **{f'val_recall_{i}': validation['mixed']['classification_report'][c]['recall']
                               for i, c in enumerate(CLASSES)}, 'seconds': time.perf_counter() - begin})
            write_csv(folder / 'history.csv', history)
            print(f'seed={seed} epoch={n} train_loss={tl:.4f} val_loss={vl:.4f} val_acc={va:.3f} val_f1={validation['mixed']['macro_f1']:.3f} seconds={history[-1]["seconds"]:.1f}', flush=True)
            if vl < best:
                best, stale, best_epoch = vl, 0, n
                torch.save({'state_dict': model.state_dict(), 'head': args.head, 'dropout': args.dropout,
                            'classes': CLASSES, 'seed': seed, 'best_epoch': n}, folder / 'best.pt')
                dump(folder / 'validation_metrics.json', {'seed': seed, 'best_epoch': n,
                     'selection': 'minimum unweighted validation cross entropy', 'loss': vl,
                     'scenarios': validation})
                save_predictions(folder / 'validation_predictions.csv', subsets['validation'], vp)
            else:
                stale += 1
            # This checkpoint is independent of loss selection and does not reset patience.
            current_f1 = validation['mixed']['macro_f1']
            if current_f1 > best_f1:
                best_f1, best_f1_epoch = current_f1, n
                torch.save({'state_dict': model.state_dict(), 'head': args.head, 'dropout': args.dropout,
                            'classes': CLASSES, 'seed': seed, 'best_epoch': n,
                            'selection': 'maximum mixed validation macro F1',
                            'validation_macro_f1': current_f1}, folder / 'best_macro_f1.pt')
                dump(folder / 'validation_metrics_macro_f1.json', {'seed': seed, 'best_epoch': n,
                     'selection': 'maximum mixed validation macro F1; earliest epoch wins ties',
                     'loss': vl, 'scenarios': validation})
                save_predictions(folder / 'validation_predictions_macro_f1.csv', subsets['validation'], vp)
            if args.patience and stale >= args.patience:
                break
        dump(folder / 'checkpoint_summary.json', {
             'seed': seed, 'epochs_run': len(history),
             'minimum_loss': {'checkpoint': 'best.pt', 'epoch': best_epoch, 'value': best},
             'maximum_macro_f1': {'checkpoint': 'best_macro_f1.pt', 'epoch': best_f1_epoch, 'value': best_f1},
             'early_stopping_metric': 'unweighted validation cross entropy',
             'test_checkpoint': None if args.validation_only else 'best.pt'})
        print(f'seed={seed} selected loss epoch={best_epoch}; macro F1 epoch={best_f1_epoch} (F1={best_f1:.4f})', flush=True)
        model.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
        if args.validation_only:
            dump(folder / 'training_complete.json', {'seed': seed, 'best_epoch': best_epoch,
                 'epochs_run': len(history), 'seconds': time.perf_counter() - started,
                 'test_evaluated': False})
            plot_outputs(folder, history, {})
            continue
        _, _, y, pred = epoch(model, batches['test'])
        results = {}
        for background in ('mixed', 'field', 'white'):
            indices = [i for i, r in enumerate(subsets['test']) if background == 'mixed' or r['background'] == background]
            results[background] = metrics([y[i] for i in indices], [pred[i] for i in indices])
        predictions = [{**r, 'true_class': CLASSES[r['label']], 'predicted_class': CLASSES[p], 'correct': r['label'] == p}
                       for r, p in zip(subsets['test'], pred)]
        write_csv(folder / 'test_predictions.csv', predictions)
        dump(folder / 'metrics.json', {'seed': seed, 'best_epoch': best_epoch, 'epochs_run': len(history),
                                     'seconds': time.perf_counter() - started, 'scenarios': results})
        plot_outputs(folder, history, results)
    print(f'Completed {len(args.seeds)} splits in {experiment}.' +
          (' Validation-only: test data was not evaluated.' if args.validation_only else ' Run summarize next.'))


def summarize(args):
    folder = Path(args.data).resolve() / WORK / 'runs' / args.run
    config = json.loads((folder / 'config.json').read_text())
    results = [json.loads((folder / f'seed_{s}' / 'metrics.json').read_text()) for s in config['seeds']]
    if len(results) != 5 or len(set(config['seeds'])) != 5:
        raise ValueError('Assignment summary needs exactly five completed distinct seeds. Smoke runs are not report results.')
    rows = []
    for bg in ('white', 'field', 'mixed'):
        row = {'scenario': bg, 'runs': 5}
        for key in ('accuracy', 'macro_f1'):
            values = [r['scenarios'][bg][key] for r in results]
            row[key + '_mean'] = float(np.mean(values))
            row[key + '_sample_sd'] = float(np.std(values, ddof=1))
        rows.append(row)
    write_csv(folder / 'summary.csv', rows)
    dump(folder / 'summary.json', rows)
    print(json.dumps(rows, indent=2))
    print('Metrics are fractions; multiply by 100 for percentages. SD uses ddof=1.')


def predict(args):
    torch, _, _ = torch_modules()
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = make_model(checkpoint['head'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    with Image.open(args.image) as im:
        x = transforms_for(False, False)(ImageOps.exif_transpose(im).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(x).softmax(1)[0].tolist()
    print(json.dumps({'prediction': CLASSES[int(np.argmax(probabilities))],
                      'softmax_scores': dict(zip(CLASSES, probabilities))}, indent=2))



def scenario_metrics(rows, y, pred):
    results = {}
    for background in ('mixed', 'field', 'white'):
        indices = [i for i, r in enumerate(rows) if background == 'mixed' or r['background'] == background]
        if not indices:
            raise ValueError(f'No {background} evaluation images.')
        results[background] = metrics([y[i] for i in indices], [pred[i] for i in indices])
    return results


def save_predictions(path, rows, predictions):
    write_csv(path, [{**r, 'true_class': CLASSES[r['label']], 'predicted_class': CLASSES[p],
                      'correct': r['label'] == p} for r, p in zip(rows, predictions)])


def preview(args):
    # Sample only the specified split's training partition, never validation/test.
    torch, _, _ = torch_modules()
    root = Path(args.data).resolve()
    allocated = splits(load_rows(root, args.manifest), args.seed)
    candidates = [r for r in allocated if r['split'] == 'train']
    folder = root / WORK / 'previews' / args.run
    if folder.exists():
        raise ValueError(f'Preview already exists: {folder}. Use a new --run name.')
    rng = random.Random(args.seed)
    selected = []
    for label in range(5):
        for bg in ('field', 'white'):
            group = [r for r in candidates if r['label'] == label and r['background'] == bg]
            selected.extend(rng.sample(group, min(args.per_stratum, len(group))))
    folder.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    def visible(t):
        return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    manifest = []
    for i, r in enumerate(selected):
        with Image.open(root / r['path']) as im:
            original = ImageOps.exif_transpose(im).convert('RGB')
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))
        axes[0].imshow(original); axes[0].set_title('Original (aspect preserved)')
        axes[1].imshow(visible(transforms_for(False, False)(original))); axes[1].set_title('224 x 224 input')
        for j in range(2, 5):
            axes[j].imshow(visible(transforms_for(True, True)(original)))
            axes[j].set_title(f'Augmentation {j-1}')
        for ax in axes:
            ax.axis('off')
        fig.suptitle(f"{CLASSES[r['label']]} | {r['background']} | {r['path']}", fontsize=9)
        fig.tight_layout(); name = f'preview_{i:02d}.png'
        fig.savefig(folder / name, dpi=120); plt.close(fig)
        manifest.append({**r, 'preview': name})
    write_csv(folder / 'selected_training_images.csv', manifest)
    dump(folder / 'config.json', {'seed': args.seed, 'classes': CLASSES,
         'source_partition': 'train', 'per_stratum': args.per_stratum,
         'normalization_inverted_for_display': True})
    print(f'{len(selected)} training-only preview sheets saved to {folder}')


def evaluate_validation(args):
    """Evaluate an existing checkpoint using its saved validation allocation."""
    torch, _, _ = torch_modules()
    torch.set_num_threads(args.threads)
    root = Path(args.data).resolve()
    folder = root / WORK / 'runs' / args.run / f'seed_{args.seed}'
    with (folder / 'splits.csv').open(newline='', encoding='utf-8') as f:
        allocated = list(csv.DictReader(f))
    rows = []
    for r in allocated:
        if r['split'] != 'validation':
            continue
        r['label'] = int(r['label'])
        path = (root / r['path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError(f"Validation file moved/changed: {r['path']}")
        rows.append(r)
    checkpoint_name = 'best.pt' if args.checkpoint_choice == 'loss' else 'best_macro_f1.pt'
    checkpoint = torch.load(folder / checkpoint_name, map_location='cpu', weights_only=True)
    if checkpoint['classes'] != CLASSES or checkpoint['seed'] != args.seed:
        raise ValueError('Checkpoint class order or seed does not match.')
    model = make_model(checkpoint['head'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    loss, _, y, pred = epoch(model, loader(root, rows, args.batch_size))
    result = {'seed': args.seed, 'best_epoch': checkpoint['best_epoch'],
              'checkpoint': checkpoint_name, 'loss': loss, 'scenarios': scenario_metrics(rows, y, pred)}
    # Preserve baseline files: use a separate review directory, refusing overwrites.
    out = folder / ('validation_review' if args.checkpoint_choice == 'loss' else 'validation_review_macro_f1')
    if out.exists():
        raise ValueError(f'Validation review already exists: {out}. Read its metrics.json.')
    dump(out / 'metrics.json', result)
    save_predictions(out / 'predictions.csv', rows, pred)
    print(json.dumps(result, indent=2))
    print(f'Validation only; baseline checkpoint and test outputs unchanged. Saved to {out}')


def evaluate_saved(args):
    """Evaluate fixed saved checkpoints across runs; never fit or select on test data."""
    torch, _, _ = torch_modules()
    torch.set_num_threads(args.threads)
    root = Path(args.data).resolve()
    output = root / WORK / 'evaluations' / args.output
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Keep it and inspect its results; no files were overwritten.')
    checkpoint_name = 'best_macro_f1.pt' if args.checkpoint_choice == 'macro-f1' else 'best.pt'
    jobs, seeds, hashes, reference, inventory_reference = [], set(), {}, None, None
    # One-time protocol check before creating output or evaluating test predictions.
    keys = ['head', 'dropout', 'batch_size', 'lr', 'epochs', 'patience', 'threads',
            'no_augment', 'class_weighted', 'classes', 'split_proportions',
            'weights_sha256', 'versions', 'checkpoint_rules', 'stopping_rule']
    for run in args.runs:
        folder = root / WORK / 'runs' / run
        config = json.loads((folder / 'config.json').read_text())
        signature = {k: config.get(k) for k in keys}
        if reference is not None and signature != reference:
            raise ValueError(f'Experiment settings differ across source runs: {run}')
        reference = signature
        fingerprint = config.get('dataset_inventory_sha256')
        if not fingerprint or (inventory_reference is not None and fingerprint != inventory_reference):
            raise ValueError('Source dataset inventory fingerprints are missing or differ.')
        inventory_reference = fingerprint
        if bool(config.get('class_weighted', False)) != (args.expected_weighting == 'weighted'):
            raise ValueError(f'{run}: class weighting does not match --expected-weighting.')
        if config.get('classes') != CLASSES:
            raise ValueError(f'{run}: class order does not match this implementation.')
        for seed in config['seeds']:
            if seed in seeds:
                raise ValueError(f'Seed {seed} occurs more than once. Supply disjoint source runs.')
            seeds.add(seed)
            seed_folder = folder / f'seed_{seed}'
            if not (seed_folder / 'training_complete.json').exists() and not (seed_folder / 'metrics.json').exists():
                raise ValueError(f'Training is incomplete: {seed_folder}')
            with (seed_folder / 'splits.csv').open(newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError('Empty saved split.')
            paths, digest_splits = set(), {}
            for r in rows:
                r['label'] = int(r['label'])
                if not 0 <= r['label'] < 5 or r['background'] not in ('field', 'white') or r['split'] not in ('train', 'validation', 'test'):
                    raise ValueError('Invalid saved split label, background or partition.')
                file = (root / r['path']).resolve()
                if not file.is_relative_to(root) or file in paths or WORK in file.relative_to(root).parts:
                    raise ValueError(f'Unsafe or repeated image path: {r["path"]}')
                paths.add(file)
                if file not in hashes:
                    hashes[file] = hashlib.sha256(file.read_bytes()).hexdigest()
                if hashes[file] != r['sha256']:
                    raise ValueError(f'Image changed since training: {r["path"]}')
                prior = digest_splits.setdefault(r['sha256'], r['split'])
                if prior != r['split']:
                    raise ValueError('An identical image crosses saved split boundaries.')
            # Recreate the original allocation from the stored image inventory to detect edited labels/splits.
            base_rows = [{k: r[k] for k in ('path', 'label', 'background', 'sha256')} for r in rows]
            if hashlib.sha256(json.dumps(base_rows, sort_keys=True).encode()).hexdigest() != fingerprint:
                raise ValueError(f'{run}, seed {seed}: saved inventory does not match training config.')
            if splits(base_rows, seed) != rows:
                raise ValueError(f'{run}, seed {seed}: saved split allocation differs from training protocol.')
            test_rows = [r for r in rows if r['split'] == 'test']
            if {(r['label'], r['background']) for r in test_rows} != {(i, bg) for i in range(5) for bg in ('field', 'white')}:
                raise ValueError('Test split must contain every class/background combination.')
            file = seed_folder / checkpoint_name
            checkpoint = torch.load(file, map_location='cpu', weights_only=True)
            if checkpoint['seed'] != seed or checkpoint['classes'] != CLASSES:
                raise ValueError(f'Checkpoint seed/classes mismatch: {file}')
            if checkpoint['head'] != config['head'] or checkpoint['dropout'] != config['dropout']:
                raise ValueError('Checkpoint architecture differs from source config.')
            summary = json.loads((seed_folder / 'checkpoint_summary.json').read_text())
            selection = summary['maximum_macro_f1' if args.checkpoint_choice == 'macro-f1' else 'minimum_loss']
            if checkpoint['best_epoch'] != selection['epoch'] or selection['checkpoint'] != checkpoint_name:
                raise ValueError('Checkpoint metadata differs from saved selection summary.')
            # Strict loading also checks tensor keys/shapes before test evaluation.
            model = make_model(checkpoint['head'], checkpoint['dropout'])
            model.load_state_dict(checkpoint['state_dict'])
            jobs.append({'seed': seed, 'run': run, 'checkpoint': file, 'test_rows': test_rows,
                         'checkpoint_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                         'best_epoch': checkpoint['best_epoch']})
            del model, checkpoint
    if seeds != set(args.expected_seeds):
        raise ValueError(f'Expected seeds {args.expected_seeds}; found {sorted(seeds)}.')
    jobs.sort(key=lambda j: j['seed'])
    print(f'Checks passed: {len(jobs)} completed splits; weighting={args.expected_weighting}; checkpoint={checkpoint_name}.', flush=True)
    for job in jobs:
        print(f"  seed={job['seed']} source={job['run']} epoch={job['best_epoch']} test_images={len(job['test_rows'])}", flush=True)
    if args.check_only:
        print('Check-only complete. No test predictions were generated and no output directory was created.')
        return
    provenance = {'checkpoint_choice': args.checkpoint_choice, 'expected_weighting': args.expected_weighting,
                  'source_runs': args.runs, 'seeds': sorted(seeds), 'source_settings': reference,
                  'dataset_inventory_sha256': inventory_reference,
                  'evaluation_versions': {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'numpy', 'scikit-learn')},
                  'selection_basis': 'Fixed from validation comparison before this test evaluation',
                  'limitation': 'Original minimum-loss baseline test results were inspected during development; this is not an untouched final holdout.',
                  'sources': [{k: str(v) if isinstance(v, Path) else v for k, v in job.items() if k != 'test_rows'} for job in jobs]}
    dump(output / 'provenance.json', provenance)
    results, per_run = [], []
    for job in jobs:
        file = job['checkpoint']
        if hashlib.sha256(file.read_bytes()).hexdigest() != job['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after preflight.')
        checkpoint = torch.load(file, map_location='cpu', weights_only=True)
        model = make_model(checkpoint['head'], checkpoint['dropout'])
        model.load_state_dict(checkpoint['state_dict'])
        started = time.perf_counter()
        _, _, y, pred = epoch(model, loader(root, job['test_rows'], args.batch_size))
        scenarios = scenario_metrics(job['test_rows'], y, pred)
        result = {'seed': job['seed'], 'source_run': job['run'], 'best_epoch': job['best_epoch'],
                  'checkpoint': checkpoint_name, 'seconds': time.perf_counter() - started, 'scenarios': scenarios}
        folder = output / f"seed_{job['seed']}"
        dump(folder / 'metrics.json', result)
        save_predictions(folder / 'test_predictions.csv', job['test_rows'], pred)
        # Preserve the original training curves, copied as fresh plots in the evaluation directory.
        with (file.parent / 'history.csv').open(newline='', encoding='utf-8') as f:
            history = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        plot_outputs(folder, history, scenarios)
        results.append(result)
        for bg, r in scenarios.items():
            per_run.append({'seed': job['seed'], 'source_run': job['run'], 'epoch': job['best_epoch'],
                            'scenario': bg, 'n': r['n'], 'accuracy': r['accuracy'], 'macro_f1': r['macro_f1']})
        print(f"seed={job['seed']} test accuracy={scenarios['mixed']['accuracy']:.4f} macro_F1={scenarios['mixed']['macro_f1']:.4f}", flush=True)
    summary_rows, class_rows = [], []
    for bg in ('white', 'field', 'mixed'):
        row = {'scenario': bg, 'runs': len(results)}
        for key in ('accuracy', 'macro_f1'):
            values = [r['scenarios'][bg][key] for r in results]
            row[key + '_mean'] = float(np.mean(values))
            row[key + '_sample_sd'] = float(np.std(values, ddof=1))
        summary_rows.append(row)
        for c in CLASSES:
            cr = {'scenario': bg, 'class': c, 'runs': len(results)}
            for key in ('precision', 'recall', 'f1-score'):
                values = [r['scenarios'][bg]['classification_report'][c][key] for r in results]
                cr[key + '_mean'] = float(np.mean(values))
                cr[key + '_sample_sd'] = float(np.std(values, ddof=1))
            class_rows.append(cr)
    write_csv(output / 'per_seed_metrics.csv', per_run)
    write_csv(output / 'summary.csv', summary_rows)
    write_csv(output / 'per_class_summary.csv', class_rows)
    dump(output / 'summary.json', summary_rows)
    dump(output / 'complete.json', {'seeds': sorted(seeds), 'runs': len(results), 'complete': True})
    print(json.dumps(summary_rows, indent=2))
    print(f'Final evaluation saved to {output}. Metrics are fractions; SD uses ddof=1. No training was performed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, func in [('audit', audit), ('download-weights', download), ('train', train), ('summarize', summarize), ('preview', preview), ('evaluate-validation', evaluate_validation)]:
        p = commands.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument('--data', default='./Dhan-Shomadhan')
        if name in ('audit', 'train', 'preview'):
            p.add_argument('--manifest', help='Optional CSV with path,class,background columns')
        if name in ('train', 'summarize', 'preview', 'evaluate-validation'):
            p.add_argument('--run', required=True, help='New experiment name, e.g. smoke or paper_50')
        if name in ('preview', 'evaluate-validation'):
            p.add_argument('--seed', type=int, default=42)
        if name == 'preview':
            p.add_argument('--per-stratum', type=int, default=1)
        if name == 'evaluate-validation':
            p.add_argument('--checkpoint-choice', choices=['loss', 'macro-f1'], default='loss')
            p.add_argument('--threads', type=int, default=4)
            p.add_argument('--batch-size', type=int, default=16)
        if name == 'train':
            p.add_argument('--head', choices=['paper', 'pooled'], default='paper')
            p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
            p.add_argument('--epochs', type=int, default=400)
            p.add_argument('--patience', type=int, default=0, help='0 disables early stopping')
            p.add_argument('--batch-size', type=int, default=16)
            p.add_argument('--lr', type=float, default=.0001)
            p.add_argument('--dropout', type=float, default=.5, help='Paper rate unspecified: declared assumption')
            p.add_argument('--threads', type=int, default=4)
            p.add_argument('--no-augment', action='store_true')
            p.add_argument('--class-weighted', action='store_true', help='Weight training loss by inverse training class frequency')
            p.add_argument('--validation-only', action='store_true', help='Skip test evaluation for development pilots')
    p = commands.add_parser('evaluate-saved', help='Evaluate five saved checkpoints across disjoint source runs')
    p.set_defaults(func=evaluate_saved)
    p.add_argument('--data', default='./Dhan-Shomadhan')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint-choice', choices=['loss', 'macro-f1'], default='macro-f1')
    p.add_argument('--expected-weighting', choices=['unweighted', 'weighted'], default='unweighted')
    p.add_argument('--expected-seeds', nargs=5, type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--check-only', action='store_true')
    p = commands.add_parser('predict'); p.set_defaults(func=predict)
    p.add_argument('--checkpoint', required=True); p.add_argument('--image', required=True)
    p.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if hasattr(args, 'run') and (not re.fullmatch(r'[A-Za-z0-9_-]+', args.run)):
        parser.error('--run must contain only letters, numbers, underscores or hyphens.')
    if args.command == 'evaluate-saved':
        if not all(re.fullmatch(r'[A-Za-z0-9_-]+', n) for n in args.runs + [args.output]):
            parser.error('Source run and output names must use letters, numbers, underscores or hyphens.')
        if len(set(args.expected_seeds)) != 5 or args.threads < 1 or args.batch_size < 1:
            parser.error('Use five distinct expected seeds and positive threads/batch size.')
    if hasattr(args, 'seed') and not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 2**32-1.')
    if args.command == 'preview' and args.per_stratum < 1:
        parser.error('--per-stratum must be positive.')
    if args.command == 'evaluate-validation' and (args.threads < 1 or args.batch_size < 1):
        parser.error('Threads and batch size must be positive.')
    if args.command == 'train':
        if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.lr <= 0 or not 0 <= args.dropout < 1 or args.patience < 0:
            parser.error('Invalid training hyperparameters.')
        if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
            parser.error('Use distinct seeds between 0 and 2**32-1.')
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
