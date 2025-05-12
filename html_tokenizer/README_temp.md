Download the parquet files from:

https://huggingface.co/datasets/HuggingFaceM4/WebSight/viewer/v0.2/train

Add the parquet files to an input_dir of your choice and then run the extractor:

```python
python html_tokenizer/extract_websight.py --input_dir datasets/websight/original_data --output_dir datasets/websight/all_data
```

It will output in the specified dir following the format.