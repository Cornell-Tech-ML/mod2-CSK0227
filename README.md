[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py


## Simple Dataset

50 points, x hidden layers = 8 , learning rate = 0.1, Number of Epochs = 500

```
Epoch  10  loss  38.51706206577331 correct 28
Epoch  20  loss  34.034054166399315 correct 31
Epoch  30  loss  31.636876712391093 correct 33
Epoch  40  loss  30.239099008732488 correct 33
Epoch  50  loss  29.280075363972863 correct 33
Epoch  60  loss  28.454039543535714 correct 35
Epoch  70  loss  27.749887398504534 correct 35
Epoch  80  loss  27.091321305031823 correct 35
Epoch  90  loss  26.46400822152088 correct 35
Epoch  100  loss  25.848402335542985 correct 37
Epoch  110  loss  25.246119280287314 correct 37
Epoch  120  loss  24.67618824338556 correct 37
Epoch  130  loss  24.116491781789907 correct 38
Epoch  140  loss  23.56924943838881 correct 39
Epoch  150  loss  23.023879875494526 correct 41
Epoch  160  loss  22.45995618925048 correct 41
Epoch  170  loss  21.886998675318377 correct 41
Epoch  180  loss  21.313653361732996 correct 41
Epoch  190  loss  20.750437304394435 correct 41
Epoch  200  loss  20.25114481518722 correct 40
Epoch  210  loss  19.76183896082855 correct 40
Epoch  220  loss  19.27620752591831 correct 40
Epoch  230  loss  18.79475777008149 correct 41
Epoch  240  loss  18.31918508699083 correct 42
Epoch  250  loss  17.823740908781026 correct 43
Epoch  260  loss  17.33944935008108 correct 45
Epoch  270  loss  16.858929502924198 correct 45
Epoch  280  loss  16.381590661988678 correct 45
Epoch  290  loss  15.914237517876415 correct 45
Epoch  300  loss  15.461693617507635 correct 46
Epoch  310  loss  15.01479042903628 correct 46
Epoch  320  loss  14.572949081010426 correct 46
Epoch  330  loss  14.137126983827836 correct 46
Epoch  340  loss  13.711317451371302 correct 46
Epoch  350  loss  13.2910501456954 correct 46
Epoch  360  loss  12.72956650826278 correct 46
Epoch  370  loss  12.212646967847299 correct 46
Epoch  380  loss  11.742557168345153 correct 46
Epoch  390  loss  11.301331309265107 correct 45
Epoch  400  loss  10.881540059191007 correct 46
Epoch  410  loss  10.478111342518083 correct 46
Epoch  420  loss  10.083589244792412 correct 46
Epoch  430  loss  9.701821741011617 correct 46
Epoch  440  loss  9.333516991683137 correct 46
Epoch  450  loss  8.977526596118343 correct 46
Epoch  460  loss  8.635507492333161 correct 46
Epoch  470  loss  8.311046067605098 correct 47
Epoch  480  loss  7.998693160353257 correct 48
Epoch  490  loss  7.698046662231119 correct 48
Epoch  500  loss  7.4103465926444185 correct 48
```

## Diag Dataset

50 points, x hidden layers = 8 , learning rate = 0.1, Number of Epochs = 500

```
Epoch  10  loss  29.080928900249642 correct 38
Epoch  20  loss  17.15711880600658 correct 43
Epoch  30  loss  12.431204950629319 correct 45
Epoch  40  loss  10.338174486287734 correct 44
Epoch  50  loss  9.107372414559398 correct 46
Epoch  60  loss  8.177031670756307 correct 46
Epoch  70  loss  7.508536887185261 correct 47
Epoch  80  loss  6.997072787897595 correct 47
Epoch  90  loss  6.520039437142539 correct 48
Epoch  100  loss  6.056121213106231 correct 48
Epoch  110  loss  5.619271201541227 correct 48
Epoch  120  loss  5.211278165794306 correct 48
Epoch  130  loss  4.829900966420983 correct 48
Epoch  140  loss  4.478856770641898 correct 49
Epoch  150  loss  4.159748538614907 correct 49
Epoch  160  loss  3.8770744625092926 correct 49
Epoch  170  loss  3.6229516318605532 correct 49
Epoch  180  loss  3.416753832501791 correct 49
Epoch  190  loss  3.224653788791028 correct 49
Epoch  200  loss  3.038070114947434 correct 49
Epoch  210  loss  2.8648258849212604 correct 49
Epoch  220  loss  2.7010186768109263 correct 49
Epoch  230  loss  2.548673409761378 correct 49
Epoch  240  loss  2.4091131588004546 correct 49
Epoch  250  loss  2.2748763211958027 correct 49
Epoch  260  loss  2.1502205499151548 correct 49
Epoch  270  loss  2.033529406676765 correct 50
Epoch  280  loss  1.9241451526009294 correct 50
Epoch  290  loss  1.8218157783593645 correct 50
Epoch  300  loss  1.7260042377898623 correct 50
Epoch  310  loss  1.6362097177430206 correct 50
Epoch  320  loss  1.5520275041628366 correct 50
Epoch  330  loss  1.4719623604336043 correct 50
Epoch  340  loss  1.3980224837551096 correct 50
Epoch  350  loss  1.3276245664427544 correct 50
Epoch  360  loss  1.2607015547629477 correct 50
Epoch  370  loss  1.1981920590656903 correct 50
Epoch  380  loss  1.139692256862488 correct 50
Epoch  390  loss  1.085550939138859 correct 50
Epoch  400  loss  1.0350076348169464 correct 50
Epoch  410  loss  0.9877228954802564 correct 50
Epoch  420  loss  0.943163636898771 correct 50
Epoch  430  loss  0.9018621634415573 correct 50
Epoch  440  loss  0.863172099729332 correct 50
Epoch  450  loss  0.8272032312194947 correct 50
Epoch  460  loss  0.7925125670999712 correct 50
Epoch  470  loss  0.7605607849152398 correct 50
Epoch  480  loss  0.7304951032109946 correct 50
Epoch  490  loss  0.702203364090057 correct 50
Epoch  500  loss  0.6753699774474378 correct 50
```

## Split Dataset

50 points, x hidden layers = 8 , learning rate = 0.1, Number of Epochs = 500

```
Epoch  10  loss  31.379358002169038 correct 32
Epoch  20  loss  28.419164928844232 correct 37
Epoch  30  loss  26.121062361219334 correct 39
Epoch  40  loss  24.334994098022477 correct 40
Epoch  50  loss  22.83068931919342 correct 41
Epoch  60  loss  21.469596689613414 correct 43
Epoch  70  loss  20.203962695859587 correct 44
Epoch  80  loss  19.041936531689462 correct 43
Epoch  90  loss  17.949526863171446 correct 43
Epoch  100  loss  16.93400123363047 correct 44
Epoch  110  loss  16.03638303529461 correct 44
Epoch  120  loss  15.2363710367652 correct 44
Epoch  130  loss  14.480516280434411 correct 44
Epoch  140  loss  13.7922019672608 correct 44
Epoch  150  loss  13.153735402827568 correct 46
Epoch  160  loss  12.561111745851413 correct 46
Epoch  170  loss  12.012341937665758 correct 46
Epoch  180  loss  11.515437038873461 correct 46
Epoch  190  loss  11.052555238053076 correct 46
Epoch  200  loss  10.626872267754266 correct 46
Epoch  210  loss  10.232916563295696 correct 46
Epoch  220  loss  9.866666921870783 correct 47
Epoch  230  loss  9.525540192742133 correct 47
Epoch  240  loss  9.205626355408548 correct 48
Epoch  250  loss  8.906584856984491 correct 48
Epoch  260  loss  8.624649219453412 correct 48
Epoch  270  loss  8.35806765063523 correct 48
Epoch  280  loss  8.105352346588683 correct 48
Epoch  290  loss  7.864955976832555 correct 48
Epoch  300  loss  7.63533434353144 correct 48
Epoch  310  loss  7.41547571805195 correct 49
Epoch  320  loss  7.2046971384425404 correct 49
Epoch  330  loss  7.001806267348015 correct 49
Epoch  340  loss  6.810569251350506 correct 49
Epoch  350  loss  6.636129828298413 correct 49
Epoch  360  loss  6.476433969526667 correct 49
Epoch  370  loss  6.328277414223695 correct 49
Epoch  380  loss  6.1921674373667 correct 49
Epoch  390  loss  6.0626607792077225 correct 49
Epoch  400  loss  5.94570886628934 correct 49
Epoch  410  loss  5.832644588908167 correct 49
Epoch  420  loss  5.726089214088112 correct 49
Epoch  430  loss  5.624014933311297 correct 49
Epoch  440  loss  5.529619801999806 correct 49
Epoch  450  loss  5.4388901294513285 correct 49
Epoch  460  loss  5.357102298231715 correct 49
Epoch  470  loss  5.276947790825386 correct 49
Epoch  480  loss  5.201876220663464 correct 49
Epoch  490  loss  5.131125676359608 correct 49
Epoch  500  loss  5.064204550057726 correct
```

# Xor Dataset

50 points, x hidden layers = 8 , learning rate = 0.1, Number of Epochs = 500

```
Epoch  10  loss  32.805353914298465 correct 29
Epoch  20  loss  31.8864774597873 correct 29
Epoch  30  loss  31.240523830912384 correct 32
Epoch  40  loss  30.67520561133351 correct 34
Epoch  50  loss  30.13759002077757 correct 34
Epoch  60  loss  29.6798091790615 correct 34
Epoch  70  loss  29.255527323987963 correct 34
Epoch  80  loss  28.828724824386587 correct 34
Epoch  90  loss  28.427843173689777 correct 33
Epoch  100  loss  28.047927112935607 correct 34
Epoch  110  loss  27.674923424173056 correct 36
Epoch  120  loss  27.316436674342217 correct 36
Epoch  130  loss  26.961749486205708 correct 36
Epoch  140  loss  26.606623327175505 correct 36
Epoch  150  loss  26.25333284890251 correct 35
Epoch  160  loss  25.90084322106248 correct 35
Epoch  170  loss  25.540286416043994 correct 35
Epoch  180  loss  25.17162972808097 correct 36
Epoch  190  loss  24.80016538848045 correct 36
Epoch  200  loss  24.427211931832343 correct 36
Epoch  210  loss  24.047640862529192 correct 36
Epoch  220  loss  23.670600899957037 correct 36
Epoch  230  loss  23.291014688966037 correct 36
Epoch  240  loss  22.90142979464766 correct 36
Epoch  250  loss  22.50350382994125 correct 37
Epoch  260  loss  22.104989652404974 correct 37
Epoch  270  loss  21.695316455367788 correct 38
Epoch  280  loss  21.283661208752182 correct 38
Epoch  290  loss  20.866468597614336 correct 40
Epoch  300  loss  20.43889017274102 correct 41
Epoch  310  loss  20.007268758319228 correct 41
Epoch  320  loss  19.583424765639453 correct 41
Epoch  330  loss  19.16985822223735 correct 42
Epoch  340  loss  18.760042717146895 correct 42
Epoch  350  loss  18.36595846241638 correct 43
Epoch  360  loss  17.983835981908747 correct 43
Epoch  370  loss  17.61767244745394 correct 43
Epoch  380  loss  17.261425495150736 correct 43
Epoch  390  loss  16.917350116058675 correct 44
Epoch  400  loss  16.539231673488562 correct 45
Epoch  410  loss  16.147131463565334 correct 45
Epoch  420  loss  15.802369107810543 correct 45
Epoch  430  loss  15.479704464294496 correct 45
Epoch  440  loss  15.16809734059009 correct 45
Epoch  450  loss  14.8663639478545 correct 45
Epoch  460  loss  14.576016033323775 correct 45
Epoch  470  loss  14.254098473781715 correct 45
Epoch  480  loss  13.921332397079228 correct 45
Epoch  490  loss  13.601279203027373 correct 44
Epoch  500  loss  13.300350692782574 correct 45
```

# Circle Dataset

50 points, x hidden layers = 8 , learning rate = 0.1, Number of Epochs = 500

```
Epoch  10  loss  26.189094574617336 correct 37
Epoch  20  loss  23.439392256680744 correct 39
Epoch  30  loss  21.626238684841468 correct 39
Epoch  40  loss  20.238759762526595 correct 40
Epoch  50  loss  19.01173610201513 correct 42
Epoch  60  loss  17.936891342355167 correct 42
Epoch  70  loss  17.025339502611985 correct 43
Epoch  80  loss  16.208708878120586 correct 43
Epoch  90  loss  15.479000204838249 correct 43
Epoch  100  loss  14.827351340418014 correct 43
Epoch  110  loss  14.199303911154983 correct 44
Epoch  120  loss  13.632786815855713 correct 44
Epoch  130  loss  13.08850360584196 correct 44
Epoch  140  loss  12.567467905108229 correct 44
Epoch  150  loss  12.065719109941494 correct 45
Epoch  160  loss  11.577598237913174 correct 45
Epoch  170  loss  11.096394226888345 correct 47
Epoch  180  loss  10.645809233783247 correct 47
Epoch  190  loss  10.207174643532849 correct 47
Epoch  200  loss  9.707766266224063 correct 47
Epoch  210  loss  9.199913564193148 correct 47
Epoch  220  loss  8.701745378592527 correct 47
Epoch  230  loss  8.22379524108592 correct 47
Epoch  240  loss  7.757245354033883 correct 47
Epoch  250  loss  7.306054026487638 correct 48
Epoch  260  loss  6.87295074451829 correct 48
Epoch  270  loss  6.45639822151158 correct 48
Epoch  280  loss  6.069972941937397 correct 49
Epoch  290  loss  5.698069162066355 correct 50
Epoch  300  loss  5.3399869645641065 correct 50
Epoch  310  loss  5.004525111840157 correct 50
Epoch  320  loss  4.683168824237635 correct 50
Epoch  330  loss  4.384071812400666 correct 50
Epoch  340  loss  4.099206919046421 correct 50
Epoch  350  loss  3.834680599055155 correct 50
Epoch  360  loss  3.5895112547926327 correct 50
Epoch  370  loss  3.3522275179140535 correct 50
Epoch  380  loss  3.1345016911472876 correct 50
Epoch  390  loss  2.937517430136802 correct 50
Epoch  400  loss  2.7521954419525336 correct 50
Epoch  410  loss  2.575074595564991 correct 50
Epoch  420  loss  2.414458494690547 correct 50
Epoch  430  loss  2.2618414568310166 correct 50
Epoch  440  loss  2.113646546803519 correct 50
Epoch  450  loss  1.9734005033805153 correct 50
Epoch  460  loss  1.8454930856639467 correct 50
Epoch  470  loss  1.728987671298821 correct 50
Epoch  480  loss  1.6232213821481352 correct 50
Epoch  490  loss  1.5285462336495814 correct 50
Epoch  500  loss  1.4415875633050559 correct 50
```

# Spiral Dataset

50 points, x hidden layers = 8 , learning rate = 0.1, Number of Epochs = 500

```
Epoch  10  loss  38.352873490787346 correct 25
Epoch  20  loss  36.540572716694854 correct 24
Epoch  30  loss  35.32071848791838 correct 26
Epoch  40  loss  34.44870643906182 correct 27
Epoch  50  loss  33.852559082047755 correct 25
Epoch  60  loss  33.362515065684306 correct 26
Epoch  70  loss  32.923950025778694 correct 29
Epoch  80  loss  32.54250260099258 correct 29
Epoch  90  loss  32.16038780887875 correct 29
Epoch  100  loss  31.77253575877313 correct 29
Epoch  110  loss  31.374082309504654 correct 29
Epoch  120  loss  30.98325579323874 correct 30
Epoch  130  loss  30.627435926774425 correct 30
Epoch  140  loss  30.25511522853017 correct 31
Epoch  150  loss  29.86181040949673 correct 32
Epoch  160  loss  29.443542988545328 correct 32
Epoch  170  loss  29.00106489623283 correct 32
Epoch  180  loss  28.52855722502725 correct 32
Epoch  190  loss  28.032507503320765 correct 32
Epoch  200  loss  27.506579354578083 correct 32
Epoch  210  loss  26.95213872042774 correct 31
Epoch  220  loss  26.354142771810547 correct 33
Epoch  230  loss  25.640634456822934 correct 35
Epoch  240  loss  24.871779353835336 correct 35
Epoch  250  loss  23.98884571380259 correct 37
Epoch  260  loss  23.049143548670525 correct 38
Epoch  270  loss  22.020156630553362 correct 42
Epoch  280  loss  20.749239754670366 correct 43
Epoch  290  loss  19.61497048443437 correct 43
Epoch  300  loss  18.521849911597172 correct 43
Epoch  310  loss  17.42464515031461 correct 43
Epoch  320  loss  16.262348045574672 correct 42
Epoch  330  loss  15.159469904889312 correct 44
Epoch  340  loss  14.024584810325951 correct 46
Epoch  350  loss  13.050543953072806 correct 46
Epoch  360  loss  12.210265937520463 correct 46
Epoch  370  loss  11.434210010994441 correct 47
Epoch  380  loss  10.717239856248323 correct 47
Epoch  390  loss  10.098478083158009 correct 47
Epoch  400  loss  9.563935027482671 correct 47
Epoch  410  loss  9.088036042604609 correct 47
Epoch  420  loss  8.631582541128621 correct 47
Epoch  430  loss  8.220206892790223 correct 47
Epoch  440  loss  7.8541946286513715 correct 47
Epoch  450  loss  7.51115364279959 correct 47
Epoch  460  loss  7.205383382618403 correct 47
Epoch  470  loss  6.931982853833097 correct 48
Epoch  480  loss  6.6849769746706365 correct 48
Epoch  490  loss  6.459837815687257 correct 48
Epoch  500  loss  6.253085679718371 correct 47
```

## End