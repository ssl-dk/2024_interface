# 実験結果

骨格抽出モデルパターンと類似度検索結果は以下のようになる。

- [movenet singlepose lightning v4](#movenet-singlepose-lightning-v4)
- [movenet singlepose lightning v4 with DAMO YOLO](#movenet-singlepose-lightning-v4-with-damo-yolo)
- [movenet singlepose thunder v4](#movenet-singlepose-thunder-v4)
- [movenet singlepose thunder v4 with DAMO YOLO](#movenet-singlepose-thunder-v4-with-damo-yolo)
- [Lite HRNet 18-coco-Nx256x192 with DAMO YOLO](#lite-hrnet-18-coco-nx256x192-with-damo-yolo)
- [HRNet coco-w48-384x288 with DAMO YOLO](#hrnet-coco-w48-384x288-with-damo-yolo)


## movenet singlepose lightning v4

### query A_1.csv
| label | similarity | file_path |
| - | - | - |
| A | 0.7450044234012938 | 00_lightning_csv\A_2.csv |
| A | 0.4521558599142875 | 00_lightning_csv\A_4.csv |
| D | 0.29694326155269396 | 00_lightning_csv\D_5.csv |
| A | 0.20545524107553953 | 00_lightning_csv\A_3.csv |
| B | 0.19316869080360857 | 00_lightning_csv\B_2.csv |
| B | 0.05426617801114495 | 00_lightning_csv\B_4.csv |
| D | 0.024470875058635266 | 00_lightning_csv\D_2.csv |
| D | 0.008471934158318227 | 00_lightning_csv\D_4.csv |
| A | -0.011680579291395527 | 00_lightning_csv\A_5.csv |
| D | -0.04707788577775721 | 00_lightning_csv\D_3.csv |

### query B_1.csv
| label | similarity | file_path |
| - | - | - |
| B | 0.4543479377306008 | 00_lightning_csv\B_3.csv |
| B | 0.23489510846209916 | 00_lightning_csv\B_5.csv |
| B | 0.18162339445397013 | 00_lightning_csv\B_2.csv |
| B | 0.13546177731949316 | 00_lightning_csv\B_4.csv |
| D | 0.10349923809110852 | 00_lightning_csv\D_4.csv |
| A | -0.0013485554207322906 | 00_lightning_csv\A_4.csv |
| A | -0.0036335044391964384 | 00_lightning_csv\A_5.csv |
| A | -0.08297284806987001 | 00_lightning_csv\A_2.csv |
| D | -0.10445021831607484 | 00_lightning_csv\D_2.csv |
| A | -0.11320298122265389 | 00_lightning_csv\A_3.csv |

### query C_1.csv
| label | similarity | file_path |
| - | - | - |
| C | 0.9108300533254748 | 00_lightning_csv\C_2.csv |
| C | 0.8034199418776216 | 00_lightning_csv\C_3.csv |
| C | 0.6216801483972475 | 00_lightning_csv\C_4.csv |
| C | 0.5965247219794396 | 00_lightning_csv\C_5.csv |
| A | 0.2741127598497697 | 00_lightning_csv\A_3.csv |
| D | 0.250372576967197 | 00_lightning_csv\D_5.csv |
| D | 0.16004847719141393 | 00_lightning_csv\D_3.csv |
| A | -0.00035840906811931267 | 00_lightning_csv\A_2.csv |
| A | -0.028963283269867548 | 00_lightning_csv\A_5.csv |
| D | -0.029381693836342246 | 00_lightning_csv\D_2.csv |

### query D_1.csv
| label | similarity | file_path |
| - | - | - |
| D | 0.9114617357768244 | 00_lightning_csv\D_3.csv |
| D | 0.8862271192052609 | 00_lightning_csv\D_2.csv |
| D | 0.846887235931545 | 00_lightning_csv\D_4.csv |
| D | 0.7612968194561265 | 00_lightning_csv\D_5.csv |
| A | 0.16708833568882603 | 00_lightning_csv\A_2.csv |
| B | 0.14420142453713217 | 00_lightning_csv\B_5.csv |
| C | 0.12224328920766206 | 00_lightning_csv\C_2.csv |
| A | 0.10861760991663683 | 00_lightning_csv\A_3.csv |
| C | -0.02473215636197803 | 00_lightning_csv\C_4.csv |
| B | -0.19239304567741478 | 00_lightning_csv\B_4.csv |

## movenet singlepose lightning v4 with DAMO YOLO

### query A_1.csv
| label | similarity | file_path |
| - | - | - |
| A | 0.8412493040631932 | 00_lightning_yolo_csv\A_2.csv |
| A | 0.6939120653652432 | 00_lightning_yolo_csv\A_5.csv |
| D | 0.5553870506707196 | 00_lightning_yolo_csv\D_4.csv |
| D | 0.5414982636068438 | 00_lightning_yolo_csv\D_3.csv |
| D | 0.5049018646395969 | 00_lightning_yolo_csv\D_5.csv |
| B | 0.1548807259509694 | 00_lightning_yolo_csv\B_5.csv |
| B | 0.09557826413237443 | 00_lightning_yolo_csv\B_2.csv |
| A | 0.018196221990578826 | 00_lightning_yolo_csv\A_3.csv |
| D | -0.09488768118869871 | 00_lightning_yolo_csv\D_2.csv |
| A | -0.1329408257950815 | 00_lightning_yolo_csv\A_4.csv |

### query B_1.csv
| label | similarity | file_path |
| - | - | - |
| B | 0.8827769743131378 | 00_lightning_yolo_csv\B_4.csv |
| B | 0.6036973932528695 | 00_lightning_yolo_csv\B_3.csv |
| B | 0.5200217702917886 | 00_lightning_yolo_csv\B_5.csv |
| B | 0.40872472987900005 | 00_lightning_yolo_csv\B_2.csv |
| C | 0.34607730792101526 | 00_lightning_yolo_csv\C_4.csv |
| A | 0.2515791340120701 | 00_lightning_yolo_csv\A_4.csv |
| C | 0.24524053431035023 | 00_lightning_yolo_csv\C_5.csv |
| C | 0.2194933198905096 | 00_lightning_yolo_csv\C_3.csv |
| C | 0.1975801438926733 | 00_lightning_yolo_csv\C_2.csv |
| A | 0.13342111567773443 | 00_lightning_yolo_csv\A_3.csv |

### query C_1.csv
| label | similarity | file_path |
| - | - | - |
| C | 0.7411554084316495 | 00_lightning_yolo_csv\C_3.csv |
| C | 0.4698115251051548 | 00_lightning_yolo_csv\C_4.csv |
| A | 0.3530586338266745 | 00_lightning_yolo_csv\A_4.csv |
| C | 0.17282083920505184 | 00_lightning_yolo_csv\C_2.csv |
| C | 0.15447768032701056 | 00_lightning_yolo_csv\C_5.csv |
| B | 0.12417120811196597 | 00_lightning_yolo_csv\B_4.csv |
| D | -0.01070271911382444 | 00_lightning_yolo_csv\D_2.csv |
| B | -0.04000244086935394 | 00_lightning_yolo_csv\B_3.csv |
| B | -0.0751021372995463 | 00_lightning_yolo_csv\B_2.csv |
| D | -0.15022765853778847 | 00_lightning_yolo_csv\D_4.csv |


### query D_1.csv
| label | similarity | file_path |
| - | - | - |
| D | 0.8887378312906057 | 00_lightning_yolo_csv\D_3.csv |
| D | 0.8325112431135274 | 00_lightning_yolo_csv\D_5.csv |
| D | 0.8084970982351751 | 00_lightning_yolo_csv\D_4.csv |
| A | 0.7989769485159601 | 00_lightning_yolo_csv\A_2.csv |
| A | 0.6253446209030543 | 00_lightning_yolo_csv\A_5.csv |
| D | 0.2719193698330256 | 00_lightning_yolo_csv\D_2.csv |
| A | 0.03993543499178201 | 00_lightning_yolo_csv\A_3.csv |
| A | -0.01831820118183289 | 00_lightning_yolo_csv\A_4.csv |
| B | -0.05571538636079066 | 00_lightning_yolo_csv\B_5.csv |
| B | -0.2738804404834606 | 00_lightning_yolo_csv\B_2.csv |

## movenet singlepose thunder v4

### query A_1.csv
| label | similarity | file_path |
| - | - | - |
| A | 0.642855623767188 | 01_thunder_csv\A_4.csv |
| A | 0.5039902493053751 | 01_thunder_csv\A_5.csv |
| B | 0.4078727935403255 | 01_thunder_csv\B_4.csv |
| A | 0.334187353123609 | 01_thunder_csv\A_2.csv |
| B | 0.2937183004972806 | 01_thunder_csv\B_3.csv |
| A | 0.2877181440851806 | 01_thunder_csv\A_3.csv |
| B | 0.26061320318612646 | 01_thunder_csv\B_2.csv |
| D | 0.24008836721082086 | 01_thunder_csv\D_2.csv |
| B | 0.14790767709847766 | 01_thunder_csv\B_5.csv |
| D | 0.07357044131358752 | 01_thunder_csv\D_4.csv |

### query B_1.csv
| label | similarity | file_path |
| - | - | - |
| B | 0.7800582146204631 | 01_thunder_csv\B_5.csv |
| B | 0.720587936469061 | 01_thunder_csv\B_2.csv |
| B | 0.7136335423523658 | 01_thunder_csv\B_3.csv |
| B | 0.5626245302099525 | 01_thunder_csv\B_4.csv |
| C | 0.18616383796289487 | 01_thunder_csv\C_5.csv |
| A | -0.04472778253893819 | 01_thunder_csv\A_3.csv |
| C | -0.10204245751444298 | 01_thunder_csv\C_3.csv |
| A | -0.17132669010952067 | 01_thunder_csv\A_4.csv |
| A | -0.21007707763024758 | 01_thunder_csv\A_5.csv |
| C | -0.2437828323668169 | 01_thunder_csv\C_2.csv |

### query C_1.csv
| label | similarity | file_path |
| - | - | - |
| C | 0.884749242083639 | 01_thunder_csv\C_2.csv |
| C | 0.8809753000172573 | 01_thunder_csv\C_3.csv |
| C | 0.8758281001580858 | 01_thunder_csv\C_5.csv |
| C | 0.833559042162476 | 01_thunder_csv\C_4.csv |
| D | -0.07484375831113531 | 01_thunder_csv\D_5.csv |
| D | -0.13045091162070954 | 01_thunder_csv\D_3.csv |
| D | -0.15724835760042094 | 01_thunder_csv\D_4.csv |
| A | -0.31238824726887654 | 01_thunder_csv\A_2.csv |
| B | -0.32024927041852214 | 01_thunder_csv\B_2.csv |
| D | -0.3457705083235321 | 01_thunder_csv\D_2.csv |

### query D_1.csv
| label | similarity | file_path |
| - | - | - |
| D | 0.7400868661910402 | 01_thunder_csv\D_3.csv |
| D | 0.7086421276248502 | 01_thunder_csv\D_5.csv |
| D | 0.6730316268027247 | 01_thunder_csv\D_4.csv |
| D | 0.6309323470101341 | 01_thunder_csv\D_2.csv |
| A | 0.11614968647907362 | 01_thunder_csv\A_3.csv |
| A | 0.10055409688872959 | 01_thunder_csv\A_2.csv |
| C | 0.07593960639762025 | 01_thunder_csv\C_2.csv |
| C | -0.07780731784173271 | 01_thunder_csv\C_4.csv |
| C | -0.12183478185655404 | 01_thunder_csv\C_5.csv |
| C | -0.15365533740982534 | 01_thunder_csv\C_3.csv |

## movenet singlepose thunder v4 with DAMO YOLO
### query A_1.csv
| label | similarity | file_path |
| - | - | - |
| A | 0.6800759259204032 | 01_thunder_yolo_csv\A_2.csv |
| B | 0.6357048180255959 | 01_thunder_yolo_csv\B_2.csv |
| A | 0.5506012068059282 | 01_thunder_yolo_csv\A_4.csv |
| D | 0.4290391155650038 | 01_thunder_yolo_csv\D_5.csv |
| D | 0.4172590421879831 | 01_thunder_yolo_csv\D_2.csv |
| D | 0.3689749456593048 | 01_thunder_yolo_csv\D_4.csv |
| D | 0.34680749951629314 | 01_thunder_yolo_csv\D_3.csv |
| B | 0.22504000374178929 | 01_thunder_yolo_csv\B_4.csv |
| A | 0.15664014547948338 | 01_thunder_yolo_csv\A_3.csv |
| A | 0.05799393250763735 | 01_thunder_yolo_csv\A_5.csv |

### query B_1.csv
| label | similarity | file_path |
| - | - | - |
| B | 0.7732313919378012 | 01_thunder_yolo_csv\B_4.csv |
| B | 0.4465369279178281 | 01_thunder_yolo_csv\B_5.csv |
| A | 0.4461980011420596 | 01_thunder_yolo_csv\A_2.csv |
| B | 0.4332771128380866 | 01_thunder_yolo_csv\B_2.csv |
| B | 0.38785344372814845 | 01_thunder_yolo_csv\B_3.csv |
| D | 0.13401546902502376 | 01_thunder_yolo_csv\D_4.csv |
| D | -0.04251082155536146 | 01_thunder_yolo_csv\D_2.csv |
| C | -0.10719954353583074 | 01_thunder_yolo_csv\C_5.csv |
| C | -0.12040192244444026 | 01_thunder_yolo_csv\C_4.csv |
| D | -0.1476438658405119 | 01_thunder_yolo_csv\D_3.csv |

### query C_1.csv
| label | similarity | file_path |
| - | - | - |
| C | 0.6344531420781078 | 01_thunder_yolo_csv\C_3.csv |
| C | 0.43749910134618725 | 01_thunder_yolo_csv\C_4.csv |
| C | 0.3979339367695997 | 01_thunder_yolo_csv\C_5.csv |
| C | 0.3973629992100475 | 01_thunder_yolo_csv\C_2.csv |
| B | 0.2453100378404134 | 01_thunder_yolo_csv\B_3.csv |
| B | 0.08562405353266242 | 01_thunder_yolo_csv\B_5.csv |
| A | -0.10691439437064577 | 01_thunder_yolo_csv\A_5.csv |
| A | -0.11745142652476563 | 01_thunder_yolo_csv\A_3.csv |
| A | -0.13054681617381428 | 01_thunder_yolo_csv\A_4.csv |
| D | -0.14449421153704578 | 01_thunder_yolo_csv\D_3.csv |

### query D_1.csv
| label | similarity | file_path |
| - | - | - |
| D | 0.8806616192996827 | 01_thunder_yolo_csv\D_4.csv |
| D | 0.7931592684958668 | 01_thunder_yolo_csv\D_3.csv |
| A | 0.7192558395012123 | 01_thunder_yolo_csv\A_2.csv |
| D | 0.6857379940303718 | 01_thunder_yolo_csv\D_2.csv |
| D | 0.5194777793710672 | 01_thunder_yolo_csv\D_5.csv |
| B | 0.29690903725931866 | 01_thunder_yolo_csv\B_2.csv |
| B | 0.2370429301816397 | 01_thunder_yolo_csv\B_4.csv |
| B | -0.01775973765404392 | 01_thunder_yolo_csv\B_5.csv |
| A | -0.13165996980813693 | 01_thunder_yolo_csv\A_4.csv |
| A | -0.20249878550837952 | 01_thunder_yolo_csv\A_3.csv |

## Lite HRNet 18-coco-Nx256x192 with DAMO YOLO

### query A_1.csv
| label | similarity | file_path |
| - | - | - |
| A | 0.7708371077756366 | 02_litehrnet_yolo_csv\A_2.csv |
| A | 0.6085539564445932 | 02_litehrnet_yolo_csv\A_5.csv |
| A | 0.5799944058593797 | 02_litehrnet_yolo_csv\A_4.csv |
| D | 0.4971772711610177 | 02_litehrnet_yolo_csv\D_2.csv |
| B | 0.4914711173559497 | 02_litehrnet_yolo_csv\B_5.csv |
| A | 0.4576713799396925 | 02_litehrnet_yolo_csv\A_3.csv |
| C | 0.11042775975897919 | 02_litehrnet_yolo_csv\C_3.csv |
| B | -0.07111261791893754 | 02_litehrnet_yolo_csv\B_3.csv |
| B | -0.08676050511210337 | 02_litehrnet_yolo_csv\B_2.csv |
| D | -0.2804341610283869 | 02_litehrnet_yolo_csv\D_3.csv |

### query B_1.csv
| label | similarity | file_path |
| - | - | - |
| B | 0.827662847920267 | 02_litehrnet_yolo_csv\B_5.csv |
| B | 0.6782009531776606 | 02_litehrnet_yolo_csv\B_3.csv |
| B | 0.2612607062638308 | 02_litehrnet_yolo_csv\B_2.csv |
| A | 0.21895506128541295 | 02_litehrnet_yolo_csv\A_2.csv |
| B | 0.15807200584205205 | 02_litehrnet_yolo_csv\B_4.csv |
| A | 0.09315578688796221 | 02_litehrnet_yolo_csv\A_4.csv |
| C | 0.08831503115846896 | 02_litehrnet_yolo_csv\C_3.csv |
| A | 0.05371470644492489 | 02_litehrnet_yolo_csv\A_5.csv |
| D | 0.00785160198816209 | 02_litehrnet_yolo_csv\D_2.csv |
| C | -0.057757847592613036 | 02_litehrnet_yolo_csv\C_4.csv |

### query C_1.csv
| label | similarity | file_path |
| - | - | - |
| B | 0.817671071067601 | 02_litehrnet_yolo_csv\B_5.csv |
| A | 0.5869112673742223 | 02_litehrnet_yolo_csv\A_4.csv |
| A | 0.5044869970324416 | 02_litehrnet_yolo_csv\A_2.csv |
| A | 0.3618814587360232 | 02_litehrnet_yolo_csv\A_3.csv |
| A | 0.3572231448975123 | 02_litehrnet_yolo_csv\A_5.csv |
| C | 0.3486258034796716 | 02_litehrnet_yolo_csv\C_3.csv |
| B | 0.2914949563713127 | 02_litehrnet_yolo_csv\B_3.csv |
| D | 0.14907113544695652 | 02_litehrnet_yolo_csv\D_2.csv |
| B | 0.10248096527749853 | 02_litehrnet_yolo_csv\B_2.csv |
| C | -0.1995927235664098 | 02_litehrnet_yolo_csv\C_4.csv |

### query D_1.csv
| label | similarity | file_path |
| - | - | - |
| D | 0.8311085346023764 | 02_litehrnet_yolo_csv\D_5.csv |
| D | 0.6871352097878618 | 02_litehrnet_yolo_csv\D_3.csv |
| C | 0.5052099968168585 | 02_litehrnet_yolo_csv\C_5.csv |
| D | 0.4496334709142841 | 02_litehrnet_yolo_csv\D_4.csv |
| D | 0.4374762451646739 | 02_litehrnet_yolo_csv\D_2.csv |
| C | 0.3172398782600231 | 02_litehrnet_yolo_csv\C_2.csv |
| A | 0.17253184149391118 | 02_litehrnet_yolo_csv\A_5.csv |
| B | -0.040063586334487235 | 02_litehrnet_yolo_csv\B_4.csv |
| A | -0.04285071806199477 | 02_litehrnet_yolo_csv\A_2.csv |
| C | -0.09927910552281616 | 02_litehrnet_yolo_csv\C_4.csv |


## HRNet coco-w48-384x288 with DAMO YOLO

### query A_1.csv
| label | similarity | file_path |
| - | - | - |
| A | 0.7510546570912564 | 03_hrnet_yolo_csv\A_3.csv |
| A | 0.7315025876488851 | 03_hrnet_yolo_csv\A_5.csv |
| A | 0.6011314194124391 | 03_hrnet_yolo_csv\A_4.csv |
| C | 0.515621412189788 | 03_hrnet_yolo_csv\C_2.csv |
| D | 0.4586711312153394 | 03_hrnet_yolo_csv\D_3.csv |
| D | 0.35331604782902737 | 03_hrnet_yolo_csv\D_5.csv |
| B | 0.1106239469494117 | 03_hrnet_yolo_csv\B_3.csv |
| B | -0.0175296990647537 | 03_hrnet_yolo_csv\B_4.csv |
| B | -0.13875883921700458 | 03_hrnet_yolo_csv\B_2.csv |
| B | -0.28241953930333086 | 03_hrnet_yolo_csv\B_5.csv |

### query B_1.csv
| label | similarity | file_path |
| - | - | - |
| B | 0.7170556443536745 | 03_hrnet_yolo_csv\B_2.csv |
| B | 0.6784349427018873 | 03_hrnet_yolo_csv\B_5.csv |
| B | 0.6650122656587578 | 03_hrnet_yolo_csv\B_3.csv |
| B | 0.6200170766218759 | 03_hrnet_yolo_csv\B_4.csv |
| A | 0.3291255773198695 | 03_hrnet_yolo_csv\A_2.csv |
| A | -0.0007621924997753699 | 03_hrnet_yolo_csv\A_5.csv |
| A | -0.06414007996614343 | 03_hrnet_yolo_csv\A_4.csv |
| D | -0.11447586709706169 | 03_hrnet_yolo_csv\D_4.csv |
| C | -0.16216420606439894 | 03_hrnet_yolo_csv\C_5.csv |
| C | -0.17191471188949564 | 03_hrnet_yolo_csv\C_3.csv |

### query C_1.csv
| label | similarity | file_path |
| - | - | - |
| C | 0.829464252664682 | 03_hrnet_yolo_csv\C_2.csv |
| A | 0.5837404382460863 | 03_hrnet_yolo_csv\A_3.csv |
| A | 0.5309332631691778 | 03_hrnet_yolo_csv\A_5.csv |
| D | 0.5102134363874061 | 03_hrnet_yolo_csv\D_3.csv |
| D | 0.4972926582530695 | 03_hrnet_yolo_csv\D_5.csv |
| A | 0.06601763089796497 | 03_hrnet_yolo_csv\A_4.csv |
| C | 0.05961458478711998 | 03_hrnet_yolo_csv\C_5.csv |
| C | -0.05778639571239544 | 03_hrnet_yolo_csv\C_4.csv |
| C | -0.28174622789652176 | 03_hrnet_yolo_csv\C_3.csv |
| D | -0.3218967210029477 | 03_hrnet_yolo_csv\D_4.csv |

### query D_1.csv
| label | similarity | file_path |
| - | - | - |
| D | 0.9279014052529014 | 03_hrnet_yolo_csv\D_4.csv |
| D | 0.8735736685359151 | 03_hrnet_yolo_csv\D_2.csv |
| C | 0.5752344505911794 | 03_hrnet_yolo_csv\C_4.csv |
| C | 0.5574487089399776 | 03_hrnet_yolo_csv\C_3.csv |
| A | 0.4812844623991242 | 03_hrnet_yolo_csv\A_2.csv |
| D | 0.1901306464932973 | 03_hrnet_yolo_csv\D_5.csv |
| C | 0.09174485104800043 | 03_hrnet_yolo_csv\C_5.csv |
| D | -0.10615865317799264 | 03_hrnet_yolo_csv\D_3.csv |
| B | -0.23562830626506617 | 03_hrnet_yolo_csv\B_2.csv |
| B | -0.23627299338227922 | 03_hrnet_yolo_csv\B_5.csv |
