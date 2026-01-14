from voxelcnn.datasets import Craft3DDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch
import transformers
import os 
import typing 
from voxelcnn.datasets import MinecraftTokenizer


def latent_collate_fn(batch, encoder, device="cuda"):
    """
    batch: list of raw voxel tensors [H, W, D]
    """
    voxels = torch.stack(batch, dim=0).to(device)
    voxels = voxels.unsqueeze(dim=1)  # (B, 1, H, W, D)

    encoder.eval()
    with torch.no_grad():
        latents = encoder.encode(voxels)  # (B,C, _, _, _)

    return latents


class CycleDataset(Dataset):
    def __init__(self, original_dataset, cycle_length):
        self.original_dataset = original_dataset
        self.cycle_length = cycle_length
        self.original_length = len(original_dataset)

    def __len__(self):
        return self.cycle_length

    def __getitem__(self, idx):
        # Cycle through the original dataset
        original_idx = idx % self.original_length
        return self.original_dataset[original_idx]


class RepeatDataset(Dataset):
    def __init__(self, sample, length):
        self.sample = sample
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.sample


def get_dataloaders(config, tokenizer):
    data_loaders = {}
    for subset in ("train", "val"):  # TODO: figure out why "test" fails
        dataset = Craft3DDataset(
            config.data.data_dir,
            subset,
            tokenizer=tokenizer,
            max_samples=config.data.max_samples,
            voxel_side_len=config.data.voxel_side_len,
            air_not_air=config.data.air_not_air,
            max_active_tokens=config.data.max_active_tokens,
            rotate=config.data.rotate and (subset=='train'),
            translate=config.data.translate and (subset=='train'),
            max_translation=config.data.max_translation
        )
        if config.overfit:
            dataset = RepeatDataset(dataset[0], 20000)

        if subset == "train":
            total_samples_per_epoch = 10000
            dataset = CycleDataset(dataset, total_samples_per_epoch)

        data_loaders[subset] = DataLoader(
            dataset,
            batch_size=config.loader.global_batch_size,
            shuffle=subset == "train",
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
        )
    return data_loaders["train"], data_loaders["val"]


def mdlm_dataset(
    dataset_name,
    tokenizer,
    wrap,
    mode,
    cache_dir,
    block_size=1024,
    num_proc=len(os.sched_getaffinity(0)),
    streaming=False,
):
    if wrap:
        filename = f"{dataset_name}_{mode}_bs{block_size}_wrapped.dat"
    else:
        filename = f"{dataset_name}_{mode}_bs{block_size}_unwrapped.dat"
    _path = os.path.join(cache_dir, filename)

    if utils.fsspec_exists(_path):
        LOGGER.info(f"Loading data from: {_path}")
        return datasets.load_from_disk(_path).with_format("torch")
    LOGGER.info(f"Generating new data at: {_path}")

    crop_train = dataset_name == "text8-crop"
    if mode == "train" and crop_train:
        # double block size for sub-sampling
        block_size *= 2

    if dataset_name == "wikitext103":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir
        )
    elif dataset_name == "wikitext2":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir
        )
    elif dataset_name == "ptb":
        dataset = datasets.load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif dataset_name == "lambada":
        dataset = get_lambada_test_dataset()
    elif dataset_name == "text8":
        assert wrap
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
    elif dataset_name == "text8-crop":
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size, crop_train=True
        )
    elif dataset_name == "openwebtext-train":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[:-100000]",
            cache_dir=cache_dir,
            streaming=streaming,
        )
    elif dataset_name == "openwebtext-valid":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[-100000:]",
            cache_dir=cache_dir,
            streaming=streaming,
        )
    elif dataset_name == "scientific_papers_arxiv":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "arxiv",
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    elif dataset_name == "scientific_papers_pubmed":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "pubmed",
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    elif dataset_name == "ag_news":
        dataset = datasets.load_dataset(
            "ag_news", cache_dir=cache_dir, streaming=streaming
        )
    else:
        dataset = datasets.load_dataset(
            dataset_name, cache_dir=cache_dir, streaming=streaming
        )

    if dataset_name in ["lambada", "openwebtext-train", "openwebtext-valid"]:
        data = dataset
    else:
        data = dataset[mode]

    if dataset_name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif dataset_name == "ptb":
        detokenizer = ptb_detokenizer
    elif dataset_name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif dataset_name == "lambada":
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith("scientific_papers"):
        detokenizer = scientific_papers_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        return detok

    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    BOS = tokenizer.encode(tokenizer.bos_token)[0]

    def preprocess_and_tokenize(example):
        return tokens

    if streaming:
        tokenized_dataset = data.map(
            preprocess_and_tokenize, batched=True, desc="Tokenizing"
        )
    else:
        tokenized_dataset = data.map(
            preprocess_and_tokenize,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
    if dataset_name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns("sentence")
    elif "scientific_papers" in dataset_name:
        tokenized_dataset = tokenized_dataset.remove_columns(
            ["article", "abstract", "section_names"]
        )
    elif dataset_name == "ag_news":
        tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])
    else:
        tokenized_dataset = tokenized_dataset.remove_columns("text")

    if not wrap:
        tokenized_dataset.save_to_disk(_path)
        return tokenized_dataset.with_format("torch")

    group_texts = functools.partial(
        _group_texts, block_size=block_size, bos=BOS, eos=EOS
    )
    if streaming:
        chunked_dataset = tokenized_dataset.map(
            group_texts, batched=True, desc="Grouping"
        )
    else:
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Grouping",
        )
        chunked_dataset.save_to_disk(_path)
    chunked_dataset = chunked_dataset.with_format("torch")
    return chunked_dataset


def mdlm_dataloaders(
    config, tokenizer, skip_train=False, skip_valid=False, valid_seed=None
):
    num_gpus = torch.cuda.device_count()
    assert config.loader.global_batch_size == (
        config.loader.batch_size
        * config.trainer.num_nodes
        * num_gpus
        * config.trainer.accumulate_grad_batches
    )
    if (
        config.loader.global_batch_size
        % (num_gpus * config.trainer.accumulate_grad_batches)
        != 0
    ):
        raise ValueError(
            f"Train Batch Size {config.training.batch_size}"
            f"not divisible by {num_gpus} gpus with accumulation "
            f"{config.trainer.accumulate_grad_batches}."
        )
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
            f"Eval Batch Size for {config.eval.batch_size} "
            f"not divisible by {num_gpus}."
        )
    if skip_train:
        train_set = None
    else:
        train_set = mdlm_dataset(
            config.data.train,
            tokenizer,
            mode="train",
            wrap=config.data.wrap,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
        )

    if config.data.valid in ["text8", "lm1b", "ag_news"]:
        validation_split = "test"
    else:
        validation_split = "validation"
    if skip_valid:
        valid_set = None
    else:
        valid_set = mdlm_dataset(
            config.data.valid,
            tokenizer,
            wrap=config.data.wrap,
            mode=validation_split,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            streaming=False,
        )

    if skip_train:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=not config.data.streaming,
            persistent_workers=True,
        )
        train_loader.tokenizer = tokenizer
    if skip_valid:
        valid_loader = None
    else:
        if valid_seed is None:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            generator=generator,
        )
        # Will be used in generative perplexity calculation
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader


class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        self.characters = list("abcdefghijklmnopqrstuvwxyz ")
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[RESERVED]": 6,
            "[UNK]": 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def get_tokenizer(config):
    if config.data.tokenizer_name_or_path == "text8":
        tokenizer = Text8Tokenizer()
        breakpoint()
    elif config.data.tokenizer_name_or_path == "minecraft":
        tokenizer = MinecraftTokenizer(config, config.data.air_not_air)
    else:
        raise ValueError("Tokenizer must be minecraft or text8")

    return tokenizer
