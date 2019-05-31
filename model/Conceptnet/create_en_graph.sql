CREATE TABLE `en_graph` (
  `rel` varchar(100) DEFAULT NULL,
  `start` varchar(100) DEFAULT NULL,
  `end` varchar(100) DEFAULT NULL,
  KEY `start_index` (`start`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1