# sql 书写顺序  select->from->where->group by->having->order by
# sql 执行顺序  from->where->group by->having->select->order by

# NOTE:
      1.在select子句中只可以写常数,聚合函数以及group by后指定的聚合列.但我们看到上图中也有type列,是的,只有MySQL是支持这种写法的.其他的RDBMS并不支持.
      2.group by子句中不可以写列的别名,原因跟SQL语句的执行顺序有关.因为group by先于select执行,所以你使用别名时,系统并不会识别,自然会报错.
      3.group by聚合后的结果是无序的
      4.where子句中只可以使用条件表达式,不能使用聚合函数.事实上,只有select子句和后面马上要说的having子句可以使用聚合函数.
      5.在使用group by聚合列时,若select子句中没有使用聚合函数,其效果相当于对该列进行去重distinct.但一般并不建议这么做,会加大理解难度.

## 常用函数
abs(数值):                   求绝对值
mod(被除数, 除数):            求余数
round(数值, 需要保留小数位数):  四舍五入
concat(str1, str2):         字符串拼接
length(strs):               字符串长度
lower(strs):                字符串转小写
upper(strs):                字符串转大写
replace(对象, 需替换字符, 替换后字符):  把对象里面的需要替换字符替换成替换后字符
substring(对象, 起始位置(从1开始), 截取字符个数):  字符串截取函数
current_date:               返回当前日期(年月日),无括号
current_time:               返回当前日期(时分秒),无括号
current_timestamp:          返回当前日期(年月日时分秒),无括号
extract(元素 from 日期):     将日期中的年月日时分秒单独剥离出来 extract(year from '2019-01-01') 2019
cast(对象 as 想要转换的数据类型) 将对象转化为想要的数据类型
coalesce(参数1, 参数2, ...)   返回参数中从左边开始第一个不为null的值

like 'abc%'       以abc开头的, 任意字符长度的字符串, 包括0
like '%abc%'      包含有abc的任意长度的字符串
like '%abc'       以abc结尾的任意长度的字符串
like 'abc_'       以abc开头的, 随后只有1个字符的字符串, '_'下划线代表一个字符


## case when then 语句
case when 返回真值的表达式 then 表达式
     when 返回真值的表达式 then 表达式
     ...
     else 表达式 end

select case when type="电脑" then concat("A:", type)
        when type="键盘" then concat("B:", type)
        when type="耳机" then concat("C", type)
        else null end as product_type from cargo ;

# case when then 经典应用, 行列交换
select sum(case when type="电脑" then sale_price else 0 end) as "电脑",
       sum(case when type="键盘" then sale_price else 0 end) as "键盘" from cargo ;

select type, sum(sale_price) from sale_price group by type ;


## 表的关联
      1.行方向的关联 union, union all, intersect, intersect all
      select column1 from table1 union[union all|intersect|intersect all] select column1 from table2
      两个select语句中的列数必须相同,而且对应列的数据类型必须一致, union和intersect是去除重复行的结果, 保留全部行添加all

      2.列方向的关联 inner join, outer join, left join, right join
      select xx, xx, xx from table1 inner[left outer|right outer] join table2 on table1.xx = table2.xx ;
      inner join:       返回连接条件中两表的共有部分, 其余部分不显示
      left outer join:  左边表作为主表,右边表字段没有的为null
      right outer join: 右边表为主表,左边没有的字段为null


## 性能优化要点:
    1.<>(!=)操作无法使用索引,可以使用union all查询代替
      select id from orders where amount != 100;
      (select id from orders where amount > 100)
      union all
      (select id from orders where amount < 100 and amount > 0)

    2.innodb引擎下or无法使用组合索引
      select id, product_name from orders where mobile_no="xxxx" or user_id="xxx" ;
      (select id, product_name from orders where mobile_no="xxxx")
      union
      (select id, product_name from orders where id="xxx")

    3.in适合主表大子表小,exist适合主表小子表大


# 查询物品售价高于该种物品均价的产品
select name, type sale_price from cargo as a1
where a1.sale_price >
  (select avg(sale_price) from cargo as a2 where a1.type = a2.type) ;

# 1.窗口函数 over (partition by column1 order by column2);

# 同种商品按照价格升序排序
select name, type, sale_price, rank() over (partition by type order by sale_price) as ranking from cargo ;

# rank(稀疏的有间隔), dense_rank(密度排序无间隔) row_number()

# 累加求和
select name, type, sale_price, sum(sale_price) over(order by sale_price) as quantity from cargo ;

# 累加求和的平均值
select name, type, sale_price, avg(sale_price) over(order by sale_price rows 2 preceding) as moving_avg from cargo ;

rows 2 preceding: 包括所在行及该行的前两行参与计算
rows 2 following: 包括所在行及该行的后两行参与计算


# 2.grouping 运算符 rollup, cube, grouping sets

