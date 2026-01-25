# URDF_kitchen beta2!  
<img width="600" alt="urdf_kitchen_beta" src="docs/urdf_kitchen_beta2_banner20260101.png">  
  
URDF_kitchenは、ロボットモデルの記述方式であるURDFやMJCFの作成をサポートするPythonツール群です。  
Meshファイルにジョイントポイントを設定し、ノードで接続していく方式となっており、GUIで直感的にロボットファイルを組み立てることができます。  
普段使っているCADがURDFやMJCFの出力に対応していない場合や、手書きでXMLファイルを作成している場合に役立ちます。  
重量入力や重心設定、イナーシャ計算、パーツごとの着色などにも対応しています。  
  
Beta2版ではMeshファイルとして.stlに加え新たに.obj,.daeファイルに対応しました。  
Colliderの設定にも対応し、ColliderとしてMeshを設定できる他、立方体や円錐なども設定できるようになりました。  
おまけ機能としてURDF,MJCFを読み込むことも可能となり、既存のファイルをGUI上で調整することができます。  
    
コードがPythonであるため、AIコーディングを活用すればユーザーが自由にUI変更やデバグ、機能拡張などをすることができます。  
  
# Tools  
  
### STEP 0 -  はじめに -  "Launcher"  
<img width="200" alt="urdf_kitchen_beta" src="docs/URDF_kitchen_laucher_beta2_img1.png">  
MeshSourcer, PartsEditor, Assemblerの3つのツールを起動できるラウンチャーです。  
作業はフローはMeshSourcerでパーツの下拵えをし、PartsEditorで接続部分を設定、Assemblerで組み立てて仕上げるという流れになります。  
  
### STEP 1 -  仕込み -  "MeshSourcer"  
<img width="500" alt="urdf_kitchen_beta" src="docs/img/MeshSourcer_img2.png">  
組み立ての準備の工程として、まずお手持ちのCADでロボットの部品を動くまとまりごとにユニット化し、可動の起点を原点として出力しておきます。  
MeshSourcerはそれをサポートするツールで、下記の処理が可能です。  

- .stl, .dae, .objの読み込み  
- .stl, .dae, .objへの書き出し, 複数ファイル一括変換  
- Meshの中心点や座標軸の入れ替え  
- Meshに該当する簡易Colliderの作成(Box, Cylinder, Sphere, Capsule)  
  
### STEP 2 - 調理 - "PartsEditor"  
<img width="500" alt="urdf_kitchen_beta" src="docs/PartsEditor_beta2_img1.png">  
ユニットのMeshファイルを見ながら次のパーツを接続するジョイントポイントを設定するツールです。  
ジョイントポイントは8つまで設定でき、回転軸や色なども設定や確認ができます。  
左右対称のロボットの場合、左側のパーツさえ設定すれば右側は自動で出力できます。  
設定ファイルはパーツと対になるxmlとして保存できます。  
  
### STEP 3 - 盛り付け - "Assembler"  
<img width="500" alt="urdf_kitchen_beta" src="docs/Assembler_beta2_img1.png">  
urdfをプラモデルのように組み立てられるツールです。  
設定ファイルをまとめて読み込み、パーツ同士をノードでポチポチと接続していきます。  
また、Jointの各種パラメータも設定することができます。  
こちらも、左側だけ設定や組み立てを行えば、右側は自動で組み立てることができます。  
作業途中のファイルを保存する機能や、回転軸を確認する機能などもあります。  
Inertiaの簡易チェック機能があり、インポートしたファイルの慣性テンソルが破綻している場合には簡単に修正することができます。  
  
完成したモデルはURDF形式もしくはMJCF形式でエクスポートすることができます。  
URDFのチェックは、Garrett Johnsonが作成したブラウザツールで可能です。  
ツール内にもリンクのボタンを設置しています。  

https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/  

またMJCFのチェックはMuJoCoアプリに出力されたscene.xmlをドロップすることで動作確認ができます。  
  
### OMAKE - おまけ機能 - "Import MODEL"  

<img width="400" alt="urdf_kitchen_beta" src="docs/img/Assembler_img2.png">  

Assemblerの左上に「Imoprt MODEL」というボタンがあります。  
これは既存のURDF、SDF、MJCFファイルを読み込み、ノード形式で展開するものです。  
Githubなどで公開されているほとんどのファイルが展開できます。  
ロボット同士の外見上の合体などもできてしまいます。  
（閉リンクには十分に対応できておらず、またパラメータも対応していないものがありますので、手書きやAIコーディングなどでご対応ください。）  
  
# Install  
python 3.11 + M4 Mac、python 3.13 + Windows 11 の組み合わせで動作確認をしています。  
  
### libraryとpip  

pythonは3.11か3.13の仮想空間を作成するのがおすすめです。  
以下のライブラリを導入してください。  

```
pip install numpy  
pip install PySide6  
pip install vtk  
pip install NodeGraphQt
pip install trimesh
pip install pycollada
pip install networkx
pip install xacrodoc
```   
  
### 実行方法  
  
ターミナルやパワーシェルで、DLしたファイルがある場所にcdで移動し、  
python urdf_kitchen_Launcher.py  
でラウンチャーを起動した後、各アプリを起動します。  
urdf_kitchen_StlSourcer.py  
urdf_kitchen_PartsEditor.py  
urdf_kitchen_Assembler.py  
はpythonで直接実行することもできます。  
urdf_kitchen_utils.py, urdf_kitchen_importer.py  は他のコードと同じディレクトリにおいてください。  
  
# バグレポート  
  
絶賛バグフィックス中です。  
  
# Tutorial  

前回のバージョンのものですが、作業のフローは下記の記事にまとまっています。  
公式のチュートリアルやガイドも別途製作中です。  
  
https://qiita.com/Ninagawa123/items/c4643ca92e57c3a45efb  
  
<a href="https://qiita.com/Ninagawa123/items/c4643ca92e57c3a45efb">
  <img width="400" alt="urdf_kitchen_beta" src="docs/urdf_kitchen_banner202550406.png">  
</a>
