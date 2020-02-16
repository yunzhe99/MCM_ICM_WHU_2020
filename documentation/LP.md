| 参数 | 含义 |
| :--: | :--: |
|  s1  | PVC  |
|  s2  |  PE  |
|  s3  |  PP  |
|  s4  |  PS  |
|  s5  | PET  |
|  s6  | PUR  |
|  s7  |  PC  |

# 中国的例子

$$
s=s_1+s_2+s_3+s_4+s_5+s_6+s_7
$$

## 第一个约束

中国塑料产值：9638.36亿元（2008年数据）

中国塑料加工工业协会. http://www.cppia.com.cn/cppia1/zdbd/200942111619.htm. Accessed 16 Feb. 2020.

中国GDP：319244.6（2008年数据）（亿元）

国家数据. [http://data.stats.gov.cn/search.htm?s=2008%E5%B9%B4%E4%B8%AD%E5%9B%BDGDP](http://data.stats.gov.cn/search.htm?s=2008年中国GDP). Accessed 16 Feb. 2020.

因此
$$
\theta = 9638.36/319244.6=0.03
$$
领土面积960万平方千米

“中华人民共和国领土变化.” 维基百科，自由的百科全书, 9 Feb. 2020. *Wikipedia*, [https://zh.wikipedia.org/w/index.php?title=%E4%B8%AD%E5%8D%8E%E4%BA%BA%E6%B0%91%E5%85%B1%E5%92%8C%E5%9B%BD%E9%A2%86%E5%9C%9F%E5%8F%98%E5%8C%96&oldid=58060041](https://zh.wikipedia.org/w/index.php?title=中华人民共和国领土变化&oldid=58060041).
$$
700s_1+265s_2+325s_3+255s_4+180s_5+0s_6+180s_7\le 0.03\times9600000000000
$$

## 第二个约束

陆地水资源总量：26323.20亿立方米

国家数据. [http://data.stats.gov.cn/search.htm?s=%E9%99%86%E5%9C%B0%E6%B0%B4%E8%B5%84%E6%BA%90](http://data.stats.gov.cn/search.htm?s=陆地水资源). Accessed 16 Feb. 2020.
$$
3000s_1+1650s_2+3685s_3+6355s_4+8000s_5+0s_6+5050s_7\le0.03\times2632320000000
$$

## 第三个约束

$$
\sigma = 0.7
$$

*中国是海洋塑料最多的国家，塑料垃圾如何被排入洋流？_湃客_澎湃新闻-The Paper*. https://www.thepaper.cn/newsDetail_forward_2606687. Accessed 16 Feb. 2020.

| 塑料品种 |  N   |
| :------: | :--: |
|   PVC    | 1.41 |
|    PE    | 3.14 |
|    PP    | 3.14 |
|    PS    | 3.38 |
|   PET    | 2.29 |
|   PUR    | 0.35 |
|    PC    | 1.39 |

2005年中国碳排放74.67亿吨

中国气候变化第一次两年更新报告核心内容解读_中华人民共和国生态环境部. http://www.mee.gov.cn/ywgz/ydqhbh/wsqtkz/201904/t20190419_700369.shtml. Accessed 16 Feb. 2020.

2005年GDP 187318.9亿元

GDP均

承诺GDP均降低45%

2019年 GDP 919281.1亿元

碳排放201.55亿吨
$$
1.41\times(1.41s_1+3.14s_2+3.14s_3+3.38s_4+2.29s_5+0.35s_6+1.39s_7)\le 0.04\times20155000000000
$$

## 第四个约束

中国回收率20%

*中国是海洋塑料最多的国家，塑料垃圾如何被排入洋流？_湃客_澎湃新闻-The Paper*. https://www.thepaper.cn/newsDetail_forward_2606687. Accessed 16 Feb. 2020.

排放量242.5万立方米

塑料密度差不多为1g/cm3

也就是说1000kg/m3
$$
0.41\times(s_1+s_2+s_3+s_4+s_5+s_6+s_7)\le 2425000\times 0.2
$$

# 线性规划模型

统一以t为单位
$$
s=s_1+s_2+s_3+s_4+s_5+s_6+s_7
$$

$$
700s_1+265s_2+325s_3+255s_4+180s_5+0s_6+180s_7\le 288000000000
$$

$$
3000s_1+1650s_2+3685s_3+6355s_4+8000s_5+0s_6+5050s_7\le78969600000
$$

$$
1.9881s_1+4.4274s_2+4.4274s_3+4.7658s_4+3.2289s_5+0.4935s_6+1.9599s_7)\le 806200000000
$$

$$
0.41\times(s_1+s_2+s_3+s_4+s_5+s_6+s_7)\le 485000
$$

