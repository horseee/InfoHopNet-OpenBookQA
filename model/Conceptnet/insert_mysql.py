#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pymysql
import codecs
import config
file_name = config.root_path + 'conceptnet_en.csv'

if __name__ == "__main__":
    db = pymysql.connect("localhost", config.mysql_user, config.mysql_password, config.mysql_database)
    cursor = db.cursor()
    ids = 0
    cursor.execute("delete from en_graph")
    db.commit()
    with codecs.open(file_name, 'r', 'utf-8') as fpr:
        for line in fpr:
            relation, start, end = line.strip().split('\t')
            insert_stat = 'insert into en_graph(rel, start, end) values ("{}", "{}", "{}")'
            
            try:
                cursor.execute(insert_stat.format(relation, start, end))
            except Exception:
                print("[Error] Statement: {}",format(insert_stat.format(relation, start, end)))
            db.commit()