auto-audit
==========

* 启动: cd src && lighttpd -f fastcgi.conf
* 访问: http://hostname:port/auto-audit?url=http://xxxx.xxxx.xxx/xxxx/&content=xxxx
* 模型文件
  + model
* 词典文件
  + src/dict/template-dict.json - 特征模板
  + src/dict/white-url-dict.tsv - 网址白名单
  + src/dict/pornographic-dict.tsv - 低俗色情词列表
  + src/dict/sensitive-dict.tsv - 敏感词列表
  + src/dict/political-name-dict.tsv - 政治敏感名字列（当同时出现政治敏感名词和政治敏感动词时命中）
  + src/dict/political-verb-dict.tsv - 政治敏感动词列表（当同时出现政治敏感名词和政治敏感动词时命中）
* 配置文件
  + qmodule/segment-2.2.1/conf/qsegconf.ini
  + src/fastcgi.conf