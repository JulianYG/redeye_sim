## Example Usage
- workflow.py: ```python workflow.py --test_iter 500 --tune_iter 15000 --dest ../stats/debug.csv --g 15,30 --q 15,35```
- tune.py: ```python tune.py --iter 50000 --g 20 --q 15 --d 3```
- validate.py: ```python validate.py --iter 1000 --proto ../prototxt/train/redeye/verify.prototxt --model ../models/googlenet/snapshots/verify_iter_50000.model --f ../stats/verify.csv --q 7 --d 4```
Note the `q` and `d` in `validate` example is the quantization bits and quantization depth.
