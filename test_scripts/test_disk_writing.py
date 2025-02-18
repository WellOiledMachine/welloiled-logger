import time
from pathlib import Path

Big_string = """Lorem ipsum odor amet, consectetuer adipiscing elit. Etiam vitae class 
  viverra est nisi maecenas nam himenaeos. Efficitur natoque habitasse orci himenaeos 
  ut mus. Eu condimentum porta natoque feugiat porttitor sodales metus posuere. Turpi
  fringilla habitasse condimentum ante, efficitur mattis duis imperdiet. Hac metus
  sagittis cursus nibh consectetur. Ornare iaculis ipsum; morbi placerat tempus cras
  nec sollicitudin. Aenean porttitor interdum nibh aenean praesent lacinia nostra.
  Neque morbi semper lobortis fermentum nascetur tempus gravida.

  Habitasse mollis fames habitant mus, neque montes. Vel mi nibh tincidunt interdum; 
  mi porta diam sed aptent. Habitasse vel faucibus parturient eget volutpat odio 
  dapibus. Tempus dictum ut praesent laoreet; lectus convallis facilisi. Morbi rutrum
  eu tincidunt, felis sed porta. Magna metus suscipit egestas himenaeos nullam
  sollicitudin potenti? Habitant orci tempus habitasse faucibus facilisi. Dui ante
  bibendum ridiculus amet, venenatis dui laoreet mauris. Aliquam sit venenatis dui
  tortor scelerisque cubilia lobortis sem.
  
  Congue velit pulvinar placerat blandit elementum ipsum convallis. Egestas curabitur
  parturient justo purus integer sagittis maximus. Tortor curabitur commodo,
  sollicitudin eros arcu hendrerit. Sit duis sit consequat senectus amet vitae
  imperdiet varius. Amalesuada leo penatibus, hac integer curabitur. Mollis eros
  donec conubia ridiculus et mollis nibh. Torquent congue leo semper; iaculis ultrices
  ridiculus. Ut venenatis adipiscing facilisis mi est montes suspendisse ligula.

  Dui ex ridiculus nostra etiam ut libero consectetur. Suscipit nostra quis sit nec 
  mollis pretium; fermentum ligula eros. Nam nullam felis, ornare sollicitudin 
  conubia nostra sit. Odio libero mattis sodales adipiscing nunc blandit ultrices 
  nostra. Aelit ridiculus platea non proin, amet phasellus commodo. Facilisis ornare 
  neque scelerisque class porta hac. Nam inceptos sit dictum tortor fusce consectetur 
  rutrum. Arcu euismod nunc aliquam iaculis, dui fermentum vitae laoreet. Mollis 
  fermentum varius semper venenatis sodales. Torquent non aenean laoreet at metus 
  metus.
  
  Habitant at blandit massa faucibus taciti. Efficitur est nam sociosqu duis 
  scelerisque dui. Nulla phasellus dapibus fames amet duis sociosqu enim vulputate. 
  Maecenas primis urna neque torquent finibus pharetra condimentum. Lacus proin rutrum
  suscipit nibh ornare parturient tincidunt. Neque nulla ante laoreet, est ut rutrum. 
  Id senectus tempus elementum euismod facilisis id ac fames. Egestas nostra
  scelerisque praesent litora facilisi litora leo! Mattis blandit dignissim quis 
  pulvinar; feugiat efficitur cursus sit. Sociosqu dictumst facilisi aenean viverra
  elit adipiscing convallis.
  
  Nec facilisis justo gravida imperdiet mauris quisque gravida phasellus. Himenaeos a
  sed nisl viverra at commodo mollis erat. Anibh bibendum nullam, nisi sodales dolor. 
  Lobortis laoreet volutpat vulputate primis eu, fermentum felis molestie. Habitant 
  fusce fringilla a ex sagittis pulvinar turpis curae duis. Leo cubilia feugiat hac 
  rhoncus tellus tortor. Arcu pellentesque feugiat conubia faucibus in pharetra hac. 
  Vitae blandit nisi convallis morbi gravida porta feugiat. Ultrices velit montes 
  venenatis dignissim porttitor vehicula varius nisi.
  
  Lorem vestibulum morbi placerat lobortis sociosqu metus iaculis. Tempus sociosqu ut 
  lacinia dictum nunc convallis convallis potenti litora. Placerat placerat dictumst 
  aliquam ante phasellus inceptos aliquet ante. Condimentum rhoncus habitant class 
  ante in. Blandit varius eget lobortis curabitur eget et lacus. Ante facilisi nisi 
  at vitae ligula sodales. Fringilla congue efficitur dolor sodales congue tortor 
  malesuada hac finibus. Ligula maximus eleifend in id ultrices diam; est aenean 
  condimentum.
  
  Habitant aliquet diam sem pharetra dictum odio. Semper massa convallis blandit 
  adipiscing vestibulum et varius. Nullam sociosqu parturient non himenaeos consequat 
  leo tortor erat. Sagittis lobortis pulvinar bibendum dapibus eu metus. Ridiculus 
  neque libero posuere maximus pellentesque hac sit metus integer. Elementum sapien 
  lacus semper ut ut hendrerit eget maximus.
  
  Cursus natoque ultricies quam a nulla hendrerit tincidunt mattis. Suscipit fusce 
  aptent nam aliquet cursus sed in tristique. Sociosqu fringilla tortor sociosqu 
  fringilla velit. Massa torquent venenatis penatibus class leo euismod pretium at. 
  Netus lacus lacus erat cras litora ante gravida metus. Nibh potenti phasellus 
  vulputate quis mus pharetra. Cursus tempus sagittis aliquet natoque dignissim. 
  Lacus aliquet etiam primis congue natoque curabitur hendrerit turpis consectetur. 
  Dapibus nibh euismod faucibus aenean finibus sollicitudin.
  
  Tellus vel torquent quis in dictum feugiat. Varius dictum accumsan blandit at 
  mollis luctus etiam. Porta litora senectus scelerisque dapibus rhoncus accumsan 
  dolor. Maecenas ac egestas hac fermentum orci justo neque aptent. Fusce feugiat 
  scelerisque neque diam sagittis montes pharetra nascetur. Bibendum sollicitudin 
  cursus tincidunt adipiscing in vitae metus ac. Accumsan sagittis netus porta 
  lobortis vel class. Id habitasse habitant ultricies; porttitor pellentesque posuere 
  erat fusce."""

location = Path(__file__).parent
location = location / "big_string.txt"

with open(location, "wb", 0) as file:
    file.write(Big_string.encode("utf-8"))
    time.sleep(5)
    file.write(Big_string.encode("utf-8"))
    time.sleep(2)
