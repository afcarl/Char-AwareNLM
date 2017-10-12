Character-Aware Neural Language Models
======================================
1.train.py
--------
|    |    |     |     |     | ↑   |     |     |     |
|----|----|-----|-----|-----|-----|-----|-----|-----|
| i0 | i1 | i2  | ... | ... | i34 |     |     |     |
|    |    |     |     |     |     | ↑   |     |     |
|    | i1 | i2  | ... | ... | i34 | i35 |     |     |
|    |    | ... | ... | ... | ... | ... | ... |     |
|    |    |     |     |     |     |     |     | ↑   |
|    |    |     | i19 | ... | i34 | i35 | ... | i54 |

in LSTM, zero init always

**output : ###**

2.test.py
-------
|    |    |    |     |     | ↑   | ↑   | ↑   | ↑   |
|----|----|----|-----|-----|-----|-----|-----|-----|
| i0 | i1 | i2 | ... | ... | i34 | i35 | ... | i54 |

in LSTM, zero init always

**output ppl : 184.827** 

3.test2.py
--------
| ↑  | ↑  | ↑  | ↑   | ↑   | ↑   | ↑   |
|----|----|----|-----|-----|-----|-----|
| i0 | i1 | i2 | ... | ... | i33 | i34 |

in LSTM, init to last lstm's output

**output ppl : 140.813** 

>There is no padding before i0
