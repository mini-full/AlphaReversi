# AlphaReversi

## Run the program

```bash
pip install -r requirements.txt
python main.py
```

## Submit to `mo.zju.edu.cn`

When submitting to `momodel`, we recommend to modify line 33 in `main.py` to the below:

```python
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
```
