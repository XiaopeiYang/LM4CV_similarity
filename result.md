Model Size: ViT-B/32
|patches|  without attributes |with 64 attributes|  with 64 attributes(few-shot) |on whole classes with 64 attributes|on novel classes(2 base/4 novel)|
| ------ | -------------- | ------ |  ------ | ------ | 
| fixed | 56.02|64.45| 60.66 |62.52|
| random |59.05|48.19|61.52 | 56.92|
| image| 73.66 |75.0|73.66|58.83|


|                                                  |  images        | fixed patches | random patches|
| ------------------------------------------------ | -------------- | -------------- |  ---------- |
|on novel classes(3 novel)                         |0.843           |   0.842        |   0.842     |
|on novel classes(3 novel) without attributes      |0.896           |   0.915        |   0.894     |
|on novel classes(3 novel) with few-shot training  |0.833           |   0.831        |   0.814     |
|on all classes(3 base+3 novel)                    |0.831           |   0.839        |   0.820     |
|on novel classes(4 novel)                         |0.870           |   0.871        |   0.853     |
|on novel classes(2 novel)                         |0.965           |   0.899        |   0.890     |

