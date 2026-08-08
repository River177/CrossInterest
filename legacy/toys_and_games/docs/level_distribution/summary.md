# Categories 各层级标签分布报告

- 商品总数：**801,922**
- `层内占比`：该标签商品数 / 能到达该层级的商品数。
- `全体占比`：该标签商品数 / 全部商品数。
- 同名标签可能属于不同父路径；`label_distribution.csv` 合并同名标签，`path_distribution.csv` 按完整路径分别统计。

一个例子： Toys & Games → Games & Accessories → Casino Equipment → Poker Equipment → Poker Chip Cases & Trays → Cases 


| 名称 | 第一种 | 第二种 |
|---:|---|---:
|一级标签| Toys & Games → Games & Accessories → Casino Equipment → Poker Equipment → Poker Chip Cases | Games & Accessories
|二级标签| Toys & Games → Games & Accessories → Casino Equipment → Poker Equipment → Poker Chip Cases & Trays → Cases  | Games & Accessories → Casino Equipment


## 各层级标签数量与集中度

| 层级 | 到达该层商品数 | 占全部商品 | 标签名称数 | 完整路径节点数 | 多路径同名标签数 | Top 1 层内占比 | Top 5 层内占比 | Top 10 层内占比 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 801,922 | 100.00% | 3 | 3 | 0 | 98.13% | 100.00% | 100.00% |
| 2 | 801,922 | 100.00% | 223 | 223 | 0 | 17.03% | 54.91% | 78.79% |
| 3 | 754,906 | 94.14% | 312 | 314 | 2 | 16.54% | 39.41% | 53.37% |
| 4 | 389,855 | 48.62% | 287 | 311 | 24 | 9.31% | 32.89% | 44.04% |
| 5 | 138,766 | 17.30% | 154 | 163 | 9 | 16.81% | 43.34% | 57.80% |
| 6 | 466 | 0.06% | 6 | 6 | 0 | 28.97% | 98.93% | 100.00% |

## 第 1 层标签分布

- 到达该层的商品：**801,922**
- 标签名称数：**3**；完整路径节点数：**3**

| 排名 | 标签 | 商品数 | 层内占比 | 全体占比 | 完整路径节点数 |
|---:|---|---:|---:|---:|---:|
| 1 | Toys & Games | 786,938 | 98.13% | 98.13% | 1 |
| 2 | Remote & App Controlled Vehicles & Parts | 10,737 | 1.34% | 1.34% | 1 |
| 3 | Remote & App Controlled Vehicle Parts | 4,247 | 0.53% | 0.53% | 1 |

## 第 2 层标签分布

- 到达该层的商品：**801,922**
- 标签名称数：**223**；完整路径节点数：**223**

| 排名 | 标签 | 商品数 | 层内占比 | 全体占比 | 完整路径节点数 |
|---:|---|---:|---:|---:|---:|
| 1 | Preschool | 136,564 | 17.03% | 17.03% | 1 |
| 2 | Games & Accessories | 100,240 | 12.50% | 12.50% | 1 |
| 3 | Party Supplies | 79,979 | 9.97% | 9.97% | 1 |
| 4 | Toy Figures & Playsets | 67,950 | 8.47% | 8.47% | 1 |
| 5 | Remote & App Controlled Vehicles & Parts | 55,584 | 6.93% | 6.93% | 1 |
| 6 | Sports & Outdoor Play | 45,075 | 5.62% | 5.62% | 1 |
| 7 | Arts & Crafts | 41,267 | 5.15% | 5.15% | 1 |
| 8 | Puzzles | 39,226 | 4.89% | 4.89% | 1 |
| 9 | Novelty & Gag Toys | 38,075 | 4.75% | 4.75% | 1 |
| 10 | Dolls & Accessories | 27,852 | 3.47% | 3.47% | 1 |
| 11 | Stuffed Animals & Plush Toys | 25,974 | 3.24% | 3.24% | 1 |
| 12 | Collectible Toys | 25,668 | 3.20% | 3.20% | 1 |
| 13 | Dress Up & Pretend Play | 22,929 | 2.86% | 2.86% | 1 |
| 14 | Learning & Education | 19,542 | 2.44% | 2.44% | 1 |
| 15 | Building Toys | 17,541 | 2.19% | 2.19% | 1 |
| 16 | Baby & Toddler Toys | 11,495 | 1.43% | 1.43% | 1 |
| 17 | Remote & App Controlled Vehicle Parts | 10,737 | 1.34% | 1.34% | 1 |
| 18 | Kids' Electronics | 10,441 | 1.30% | 1.30% | 1 |
| 19 | Vehicles | 8,841 | 1.10% | 1.10% | 1 |
| 20 | Power Plant & Driveline Systems | 4,247 | 0.53% | 0.53% | 1 |

完整的 223 个标签见 `label_distribution.csv`。

## 第 3 层标签分布

- 到达该层的商品：**754,906**
- 标签名称数：**312**；完整路径节点数：**314**

| 排名 | 标签 | 商品数 | 层内占比 | 全体占比 | 完整路径节点数 |
|---:|---|---:|---:|---:|---:|
| 1 | Pre-Kindergarten Toys | 124,846 | 16.54% | 15.57% | 1 |
| 2 | Card Games | 48,580 | 6.44% | 6.06% | 1 |
| 3 | Action Figures | 45,852 | 6.07% | 5.72% | 2 |
| 4 | Decorations | 42,704 | 5.66% | 5.33% | 1 |
| 5 | Remote & App Controlled Vehicle Parts | 35,506 | 4.70% | 4.43% | 1 |
| 6 | Jigsaw Puzzles | 30,286 | 4.01% | 3.78% | 1 |
| 7 | Board Games | 26,451 | 3.50% | 3.30% | 1 |
| 8 | Party Favors | 18,423 | 2.44% | 2.30% | 1 |
| 9 | Craft Kits | 15,141 | 2.01% | 1.89% | 1 |
| 10 | Pools & Water Toys | 15,090 | 2.00% | 1.88% | 1 |
| 11 | Statues, Bobbleheads & Busts | 14,349 | 1.90% | 1.79% | 1 |
| 12 | Dress-Up Accessories | 12,561 | 1.66% | 1.57% | 1 |
| 13 | Stickers | 11,759 | 1.56% | 1.47% | 1 |
| 14 | Remote & App Controlled Vehicles | 11,647 | 1.54% | 1.45% | 1 |
| 15 | Building Sets | 11,435 | 1.51% | 1.43% | 1 |
| 16 | Dolls | 10,540 | 1.40% | 1.31% | 1 |
| 17 | Game Accessories | 10,107 | 1.34% | 1.26% | 1 |
| 18 | Toddler Toys | 10,019 | 1.33% | 1.25% | 1 |
| 19 | Plush Figures | 9,246 | 1.22% | 1.15% | 1 |
| 20 | Collectible Display & Storage | 8,358 | 1.11% | 1.04% | 1 |

完整的 312 个标签见 `label_distribution.csv`。

## 第 4 层标签分布

- 到达该层的商品：**389,855**
- 标签名称数：**287**；完整路径节点数：**311**

| 排名 | 标签 | 商品数 | 层内占比 | 全体占比 | 完整路径节点数 |
|---:|---|---:|---:|---:|---:|
| 1 | Collectible Card Games | 36,282 | 9.31% | 4.52% | 1 |
| 2 | Activity | 29,751 | 7.63% | 3.71% | 2 |
| 3 | Balloons | 24,133 | 6.19% | 3.01% | 1 |
| 4 | Pretend Play | 22,563 | 5.79% | 2.81% | 2 |
| 5 | Stuffed Animals & Toys | 15,502 | 3.98% | 1.93% | 2 |
| 6 | Banners | 10,305 | 2.64% | 1.29% | 1 |
| 7 | Multi-Item Party Favor Packs | 9,924 | 2.55% | 1.24% | 1 |
| 8 | Art & Creativity | 9,403 | 2.41% | 1.17% | 2 |
| 9 | Dice & Accessories | 6,985 | 1.79% | 0.87% | 1 |
| 10 | Jewelry | 6,837 | 1.75% | 0.85% | 2 |
| 11 | Learning | 6,735 | 1.73% | 0.84% | 2 |
| 12 | Puzzles | 6,276 | 1.61% | 0.78% | 2 |
| 13 | Statues | 6,273 | 1.61% | 0.78% | 1 |
| 14 | Card Storage & Display | 6,214 | 1.59% | 0.77% | 1 |
| 15 | Remote & App Controlled Vehicle Batteries | 6,068 | 1.56% | 0.76% | 1 |
| 16 | Standard Playing Card Decks | 5,845 | 1.50% | 0.73% | 1 |
| 17 | Wheels & Tires | 5,845 | 1.50% | 0.73% | 1 |
| 18 | Power Plant & Driveline Systems | 5,679 | 1.46% | 0.71% | 1 |
| 19 | Bobbleheads | 4,905 | 1.26% | 0.61% | 1 |
| 20 | Sorting & Stacking Toys | 4,498 | 1.15% | 0.56% | 1 |

完整的 287 个标签见 `label_distribution.csv`。

## 第 5 层标签分布

- 到达该层的商品：**138,766**
- 标签名称数：**154**；完整路径节点数：**163**

| 排名 | 标签 | 商品数 | 层内占比 | 全体占比 | 完整路径节点数 |
|---:|---|---:|---:|---:|---:|
| 1 | Singles | 23,331 | 16.81% | 2.91% | 1 |
| 2 | Role Play & Dress Up | 12,174 | 8.77% | 1.52% | 2 |
| 3 | Dolls & Dollhouses | 11,844 | 8.54% | 1.48% | 1 |
| 4 | Cars, Trucks, Boats & Planes | 7,003 | 5.05% | 0.87% | 2 |
| 5 | Toy Figure Playsets | 5,783 | 4.17% | 0.72% | 2 |
| 6 | Building Sets | 5,710 | 4.11% | 0.71% | 1 |
| 7 | Role Playing Dice | 3,983 | 2.87% | 0.50% | 1 |
| 8 | Decks & Sets | 3,971 | 2.86% | 0.50% | 1 |
| 9 | Cake Toppers | 3,272 | 2.36% | 0.41% | 1 |
| 10 | Outdoor Play | 3,142 | 2.26% | 0.39% | 2 |
| 11 | Protective Sleeves | 2,737 | 1.97% | 0.34% | 1 |
| 12 | Propellers | 2,626 | 1.89% | 0.33% | 1 |
| 13 | Kitchen Sets & Play Food | 2,521 | 1.82% | 0.31% | 2 |
| 14 | Standard Game Dice | 2,468 | 1.78% | 0.31% | 1 |
| 15 | Bracelets | 2,239 | 1.61% | 0.28% | 1 |
| 16 | Booster Packs | 1,865 | 1.34% | 0.23% | 1 |
| 17 | Servos | 1,558 | 1.12% | 0.19% | 1 |
| 18 | Necklaces | 1,523 | 1.10% | 0.19% | 1 |
| 19 | Albums | 1,478 | 1.07% | 0.18% | 1 |
| 20 | Cases | 1,445 | 1.04% | 0.18% | 1 |

完整的 154 个标签见 `label_distribution.csv`。

## 第 6 层标签分布

- 到达该层的商品：**466**
- 标签名称数：**6**；完整路径节点数：**6**

| 排名 | 标签 | 商品数 | 层内占比 | 全体占比 | 完整路径节点数 |
|---:|---|---:|---:|---:|---:|
| 1 | Boxes | 135 | 28.97% | 0.02% | 1 |
| 2 | Trays & Racks | 95 | 20.39% | 0.01% | 1 |
| 3 | Packs | 93 | 19.96% | 0.01% | 1 |
| 4 | Singles | 93 | 19.96% | 0.01% | 1 |
| 5 | Cases | 45 | 9.66% | 0.01% | 1 |
| 6 | Carousels | 5 | 1.07% | 0.00% | 1 |





