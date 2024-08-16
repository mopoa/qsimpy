// Generated from qBraid v0.7.1
OPENQASM 3.0;
include 'stdgates.inc';
// Generated from qiskit v0.24.0 file qiskit/qasm/libs/qelib1.inc

// QASM 3 equivalent definitions of qelib1.inc gates

// idle gate (identity) with length gamma*sqglen
 
// generic single qubit gate

// inverse sqrt(X)
gate sxdg a { s a; h a; s a; }

// controlled-sqrt(X)

// controlled phase rotation 
gate cu1 (lambda) a, b{  
  
  u3(0,0,lambda/2) a;
  cx a,b;
  u3(0,0,-lambda/2) b;
  cx a,b;
  u3(0,0,lambda/2) b; 
  
}

// controlled-U
gate cu3(theta,phi,lambda) c, t
{
  // implements controlled-U(theta,phi,lambda) with  target t and control c
  u3(0,0,(lambda+phi)/2) c;
  u3(0,0,(lambda-phi)/2) t;
  cx c,t;
  u3(-theta/2,0,-(phi+lambda)/2) t;
  cx c,t;
  u3(theta/2,phi,0) t;
}

// two-qubit XX rotation
// gate rxx(theta) a, b
// {
//   u3(pi/2, theta, 0) a;
//   h b;
//   cx a,b;
//   u1(-theta) b;
//   cx a,b;
//   h b;
//   ******** why type error ********
//   // u2(-pi, pi-theta) b;
// }

// two-qubit ZZ rotation
gate rzz(theta) a,b
{
  cx a,b;
  u3(0,0,theta) b;
  cx a,b;
}

// relative-phase CCX

// relative-phase 3-controlled X gate

// 3-controlled X gate

// 3-controlled sqrt(X) gate, this equals the C3X gate 
// where the CU1 rotations are -pi/8 not -pi/4

// 4-controlled X gate 

qubit[22] q;
bit[22] meas;
p(4.684618627355123) q[0];
u3(1.0863380025550022,3*pi/4,-pi/2) q[3];
h q[5];
cx q[6],q[5];
h q[5];
cu1(pi/2) q[6],q[5];
u2(0,-pi) q[5];
u2(0,0) q[7];
s q[8];
cry(4.733135208207287) q[9],q[2];
sx q[2];

// rxx gate
h q[2];
h q[3];
cx q[2],q[3];
rz(3.0727799917118745) q[3];
cx q[2],q[3];
h q[3];
h q[2];

u1(-1.0947787650860548) q[10];
cx q[7],q[10];
ry(-1.0625347409922794) q[10];
ry(-1.0625347409922794) q[7];
cx q[7],q[10];
u1(1.880176928483503) q[10];
u2(-2.8545599186475075,-pi) q[7];
u2(pi/4,-pi) q[11];
sdg q[12];
sx q[14];
cx q[8],q[14];
u2(-pi/2,-pi) q[14];
u2(-pi,0) q[8];
cx q[8],q[7];
ry(-2.64825467449729) q[7];
ry(-2.64825467449729) q[8];
cx q[8],q[7];
u1(-0.28703273494228654) q[7];
u2(-pi,pi/2) q[8];
rzz(2.222268907761786) q[8],q[14];
rx(-pi/2) q[14];
rx(-pi/2) q[8];
cx q[15],q[1];
cz q[15],q[0];
h q[15];
cu3(2.7511828987678966,3.607915524002728,2.3453197008450233) q[6],q[1];
u3(0.7173110405678399,-0.6584534837615843,1.3218962725722943) q[6];
h q[16];
cx q[13],q[16];
h q[16];
cu1(pi/2) q[13],q[16];
crx(3.9824402877397977) q[12],q[13];
u2(-1.8150665738254868,-2.074436950669676) q[12];
u2(0,pi/2) q[13];
cu1(pi/2) q[15],q[13];
h q[13];
cx q[16],q[5];
rz(5.076820509783505) q[5];
cx q[16],q[5];
cry(3.9327886215807806) q[10],q[16];
ccx q[16],q[10],q[3];
swap q[10],q[15];
ch q[10],q[15];
cy q[10],q[12];
u3(pi,0.915334138422903,-0.9153341384229026) q[10];
u2(-1.6643787200110325,0) q[15];
tdg q[3];
u3(3.020868422084181,-pi/2,0.7171195701121995) q[5];
cp(0.12608146472089976) q[6],q[13];
p(2.7035376924631196) q[13];
sx q[6];
u3(2.2305641167475034,pi/2,-pi/2) q[17];
cry(5.864966885210837) q[4],q[18];
tdg q[4];
cry(3.3401519228349494) q[1],q[4];
cx q[1],q[17];
swap q[16],q[1];
u1(-pi/4) q[1];
u3(2.409662539849075,pi/2,2.714715361296376) q[17];
h q[4];
cu1(pi/8) q[14],q[4];
cx q[19],q[11];
tdg q[11];
u1(-0.8662987490768499) q[20];
cx q[21],q[11];
t q[11];
cx q[19],q[11];
u2(0,3*pi/4) q[11];
crz(0.7203317518888239) q[18],q[11];
u2(0.6044551176832745,4.458303280933144) q[18];
crz(4.660110645421072) q[18],q[2];
u2(0,-pi/2) q[2];
rx(1.4645506429344954) q[21];
cp(0.27841736785903115) q[20],q[21];
cx q[14],q[21];
cu1(-pi/8) q[21],q[4];
cx q[14],q[21];
cu1(pi/8) q[21],q[4];
crz(4.973898923189498) q[7],q[11];
cx q[21],q[11];
cu1(-pi/8) q[11],q[4];
cx q[14],q[11];
cu1(pi/8) q[11],q[4];
cx q[21],q[11];
cu1(-pi/8) q[11],q[4];
cx q[14],q[11];
cu1(pi/8) q[11],q[4];
u2(0,0.6752251705099757) q[14];
crx(0.7232928624012734) q[18],q[21];
cry(5.935241355219743) q[16],q[21];
rx(pi/2) q[16];
rzz(0.45529085355760135) q[16],q[17];
u2(-pi/2,-pi) q[16];
u3(pi,pi/2,-pi/2) q[17];
h q[4];
cry(5.698148910708643) q[4],q[11];
u3(1.7312392479830447,-2.3448996840097154,0.8434887443495818) q[11];
cx q[4],q[11];
sdg q[11];
x q[4];
u3(0.8029459398926356,2.3085745225432603,-0.8330181310465328) q[7];
crx(4.820717485323253) q[9],q[19];
u2(2.6845301397458154,4.966942791122286) q[19];

// rxx gate
h q[8];
h q[19];
cx q[8],q[19];
rz(0.6965106857196179) q[19];
cx q[8],q[19];
h q[19];
h q[8];

u1(2.674141239639132) q[19];
cu1(pi/2) q[19],q[5];
tdg q[19];
sx q[8];
ch q[18],q[8];
crz(1.5699311213649152) q[9],q[0];
ch q[0],q[20];
cu1(pi/2) q[20],q[0];
u2(0,-pi) q[0];
cu3(4.456422858305494,1.8642439256652674,4.922000266804368) q[13],q[20];
y q[13];
cx q[13],q[16];
h q[16];
cu1(pi/2) q[13],q[16];
u2(-pi/2,pi/2) q[16];
u2(2.2637477488491164,0) q[20];
cx q[20],q[10];
ry(-0.9885381502949185) q[10];
ry(0.9885381502949185) q[20];
cx q[20],q[10];
u1(1.310924376743988) q[10];
cu1(pi/2) q[3],q[0];
u2(0,0) q[0];
cz q[12],q[0];
sdg q[12];
h q[3];
cx q[3],q[5];
rz(2.8208011259109966) q[5];
cx q[3],q[5];
h q[3];
cu1(pi/2) q[19],q[3];
h q[3];
u2(-pi,-pi/2) q[9];
cx q[2],q[9];
cx q[9],q[2];

// rxx gate
h q[7];
h q[2];
cx q[7],q[2];
rz(5.072059083852748) q[2];
cx q[7],q[2];
h q[2];
h q[7];

crz(2.4951327774590375) q[2],q[8];
rzz(3.8571743303099026) q[1],q[8];
h q[1];
cx q[11],q[1];
h q[1];
cu1(pi/2) q[11],q[1];
u2(0,-pi) q[1];
s q[11];
cu1(pi/8) q[12],q[1];
u2(0.4097337081451441,0.986536898300482) q[2];
cu1(pi/2) q[2],q[8];
cx q[12],q[8];
ccx q[7],q[18],q[14];
u3(2.758152379578812,2.101486713237165,0.8332246498203593) q[14];
u1(-0.17216825617799314) q[18];
cx q[15],q[18];
ry(1.6687183415989575) q[15];
ry(-1.6687183415989575) q[18];
cx q[15],q[18];
u2(1.9943955151778097,-pi) q[15];
u1(-2.9694243974117995) q[18];
cu3(3.515322023540707,1.5782718659095798,0.27333525938642017) q[19],q[15];
swap q[15],q[13];
u2(pi/2,pi/2) q[7];
rzz(4.062645167169655) q[7],q[16];
u2(3*pi/4,-pi/2) q[16];
rx(-pi/2) q[7];

// rxx gate
h q[3];
h q[7];
cx q[3],q[7];
rz(3.855284788590332) q[7];
cx q[3],q[7];
h q[7];
h q[3];

cu1(-pi/8) q[8],q[1];
cx q[12],q[8];
cu1(pi/8) q[8],q[1];
cx q[8],q[2];
cu1(-pi/8) q[2],q[1];
cx q[12],q[2];
cu1(pi/8) q[2],q[1];
cx q[8],q[2];
cu1(-pi/8) q[2],q[1];
cx q[12],q[2];
cy q[12],q[3];
u3(3.021489437715084,-2.0467198963737525,-2.0352326175656366) q[12];
cu1(pi/8) q[2],q[1];
u2(-2.09474368130945,-pi) q[1];
p(0.9569892645981921) q[2];
cu3(4.922132731873485,0.6435379416313662,4.6381443501044215) q[2],q[7];
s q[2];
u3(pi,0.24902701053439502,-0.24902701053439502) q[8];
u3(2.20615164388034,-pi/4,pi/2) q[9];
cx q[6],q[9];
tdg q[9];
cx q[21],q[9];
cu3(4.0731281009465174,2.5357857195377083,2.7554385418524467) q[21],q[4];
ccx q[0],q[21],q[5];
h q[0];
ccx q[14],q[10],q[0];
u3(2.494344828551492,1.4401464646541156,-2.449226212402267) q[0];
cu1(pi/2) q[10],q[20];
p(0.0031083678672225563) q[10];
cu3(0.10112592125382325,1.5470391894395652,3.5124548891798075) q[10],q[3];
u3(1.4201834616806095,3.0292731285851904,2.851541519607535) q[21];
cx q[21],q[2];
cx q[2],q[21];
u3(0.6266844764726979,-2.618023179573952,1.8646448256426469) q[2];
cx q[1],q[2];
u2(-pi/2,pi/2) q[1];
u1(-3.069064012900937) q[21];
swap q[4],q[17];
cy q[17],q[18];
ch q[14],q[17];
u3(1.3798673676473039,-2.621410786489252,3.033316288236213) q[14];
h q[18];
cx q[19],q[18];
rz(3.9818089293257306) q[18];
cx q[19],q[18];
h q[18];
cswap q[17],q[18],q[0];
u2(1.4351794146781778,2.429616625520696) q[0];
u2(-pi,0) q[17];
ry(1.2418576747663082) q[18];
u1(0.037764610810516075) q[19];
sx q[4];
cx q[11],q[4];
u3(pi,-3*pi/4,3*pi/4) q[11];
u2(0.1463391399757641,-pi) q[5];
cu1(pi/2) q[5],q[16];
u2(0,0) q[16];
cx q[16],q[19];
ry(2.353366377323213) q[16];
ry(-2.353366377323213) q[19];
cx q[16],q[19];
u2(-pi,-pi) q[16];
cy q[16],q[10];
u1(-0.037764610810516075) q[19];
cx q[7],q[5];
cu3(2.7159035835104843,0.37683326365924275,5.664166523141638) q[12],q[5];
rzz(2.8401138425343384) q[20],q[7];
sxdg q[20];
cu3(5.321280208006759,4.1126029673793,5.720342519357437) q[7],q[5];
cy q[20],q[5];
cu1(1.7881886968006497) q[8],q[3];
cswap q[3],q[18],q[12];
h q[12];
cx q[17],q[12];
rz(1.4967420150074497) q[12];
cx q[17],q[12];
u1(pi/4) q[12];
cx q[1],q[12];
tdg q[12];
cx q[17],q[12];
t q[12];
cx q[1],q[12];
u2(0,-pi/4) q[12];
sxdg q[18];
cx q[21],q[3];
cx q[3],q[21];
u3(2.3761670877918193,1.287809688112585,3.049169969521712) q[21];
u3(5.162592214757015,1.0047755882524978,-1.0047755882524978) q[8];
t q[9];
cx q[6],q[9];
u2(0,3*pi/4) q[9];
cry(0.27573751355369935) q[6],q[9];
rzz(4.027911902703024) q[6],q[9];
u2(0,-pi) q[6];
cry(2.2020998860650063) q[6],q[15];
u2(0,0) q[15];
y q[9];

// rxx gate
h q[9];
h q[4];
cx q[9],q[4];
rz(5.878228003084732) q[4];
cx q[9],q[4];
h q[4];
h q[9];

cy q[11],q[9];
u1(-1.556575257750307) q[11];
cx q[15],q[11];
ry(-1.7320098992251614) q[11];
ry(-1.7320098992251614) q[15];
cx q[15],q[11];
u1(1.556575257750307) q[11];
cz q[10],q[11];
s q[10];
u2(pi/4,-0.2970470733114037) q[11];
u2(0,0) q[15];
cp(5.080419195668465) q[4],q[13];
h q[13];
cx q[19],q[13];
rz(1.7314272300100533) q[13];
cx q[19],q[13];
cu1(pi/2) q[16],q[13];
h q[13];
swap q[13],q[16];
cx q[16],q[11];
u2(0,3*pi/4) q[11];
rx(pi/2) q[19];
cx q[5],q[11];
u1(pi/4) q[11];
cx q[20],q[11];
u1(-pi/4) q[11];
cx q[5],q[11];
u1(pi/4) q[11];
cx q[20],q[11];
u2(pi/4,3*pi/4) q[11];
cx q[16],q[11];
u2(0,3*pi/4) q[11];
u1(5.495695691996359) q[16];
u3(2.5610732298296046,-3.0764635900898787,1.9147191540385409) q[5];
cu1(pi/2) q[6],q[4];
u2(-pi/2,pi/2) q[4];
rzz(2.568763326600303) q[19],q[4];
rx(-pi/2) q[19];
cswap q[14],q[2],q[19];
cx q[14],q[0];
h q[0];
cu1(pi/2) q[14],q[0];
u1(-0.5410911670761474) q[0];
rx(pi/2) q[14];
swap q[2],q[3];
swap q[2],q[1];
u3(2.1481140627867767,2.3750815188690897,-3.023060849437968) q[1];
p(4.172100840595829) q[2];
cu3(4.095685186819329,5.26375690297141,3.50123218467502) q[2],q[0];
rx(-pi/2) q[4];
swap q[4],q[7];
u3(6.181543494428037,-1.165464188175661,1.165464188175661) q[6];
cu3(2.3746221179366835,0.24497556390096523,1.632717464139418) q[7],q[4];
u3(pi,-7*pi/8,7*pi/8) q[4];
crx(3.0104240913681877) q[7],q[20];
swap q[20],q[12];
cp(2.246204785170522) q[12],q[2];
u3(pi,pi/4,-pi/4) q[7];
ccx q[8],q[6],q[15];
cp(6.249028613334083) q[13],q[8];
rz(0.9584889129261758) q[15];
cu1(3.053812760544332) q[3],q[15];
rz(1.173219411973466) q[3];
cy q[6],q[18];
ccx q[17],q[18],q[11];
u3(2.301804010680677,1.5335536792296107,-1.5335536792296107) q[11];
rx(pi/2) q[6];
rzz(2.807019919994607) q[14],q[6];
rzz(1.7796098086433616) q[6],q[14];
rx(-pi/2) q[14];
rx(-pi/2) q[6];
crz(1.9304185457267804) q[6],q[4];
u2(0,-2.3738311904958955) q[9];
cx q[9],q[10];
cx q[10],q[9];
h q[10];
swap q[10],q[19];
ccx q[10],q[8],q[13];
u3(3.7237271964634644,4.661518257996297,5.953003268671007) q[10];
cz q[10],q[14];
h q[13];
ccx q[13],q[0],q[1];
h q[1];
ccx q[1],q[0],q[6];
cu1(pi/2) q[14],q[21];
u2(2.1216659909088635,5.7909592538456) q[19];
cu3(2.4268652964535304,2.4526125033277437,3.8591279282259388) q[15],q[19];
cu3(2.229244736992953,5.853596596167231,0.5139819104794644) q[5],q[13];
h q[5];
ccx q[13],q[1],q[5];
u2(pi/4,-pi) q[13];
u3(pi,pi/2,-pi/2) q[5];
cx q[8],q[16];
cu1(2.4336542677930972) q[15],q[8];
y q[15];
swap q[19],q[16];
t q[19];
cu1(0.61081352930315) q[2],q[16];
y q[16];
cz q[1],q[16];
u1(1.2869267265143067) q[1];
rx(pi/2) q[16];
cu1(0.5289241697664371) q[8],q[12];
u3(2.670180908034282,1.072594851891493,2.8389796574928026) q[12];
u2(0.5176218685066978,-pi) q[9];
cswap q[17],q[18],q[9];
cswap q[11],q[18],q[20];
rz(5.462350466904767) q[11];
sdg q[17];
h q[18];
cu1(pi/2) q[10],q[18];
cu1(2.8770562761123726) q[10],q[0];
p(2.1123844786954313) q[10];
cu3(0.6104921506527751,2.920677509694216,1.318390862985621) q[10],q[0];
rx(pi/2) q[10];
rzz(1.4118423025420996) q[10],q[16];
rx(-pi/2) q[10];
u2(1.7672870133319272,-pi/2) q[16];
h q[18];
ch q[14],q[18];
ccx q[19],q[15],q[11];
u2(0,0.46265649913429296) q[20];
cu1(pi/2) q[3],q[9];
cy q[4],q[17];
cu3(6.269341347561659,5.052953074472819,2.0607920876575787) q[17],q[8];
u1(-2.0443020481772436) q[17];
s q[4];
cx q[5],q[17];
ry(-2.9190282070370794) q[17];
ry(-2.9190282070370794) q[5];
cx q[5],q[17];
u1(-2.6680869322074456) q[17];
u3(2.8074237100973547,2.0885871069454582,pi/2) q[5];
cu1(pi/2) q[6],q[20];
u2(1.823024570919637,-pi) q[20];
u3(5.0398071029998945,5.511509465034263,6.064193588516642) q[6];
cry(6.254474076869609) q[7],q[3];
swap q[21],q[7];
cswap q[19],q[7],q[18];
s q[18];
rzz(6.240975456417946) q[19],q[0];
u2(0,1.927760296603962) q[0];
h q[19];
cx q[10],q[19];
rz(0.5908554232965041) q[19];
cx q[10],q[19];
u3(0.8319758496124176,0.5386662986602921,1.9028742888045755) q[19];
sx q[3];
cx q[4],q[3];
x q[4];
cp(2.0890538027827015) q[14],q[4];
rx(pi/2) q[14];
z q[4];
y q[7];
rz(3.082617189700517) q[8];
u1(-pi/4) q[9];
crx(1.9605361019834837) q[2],q[9];
cx q[2],q[13];
u2(0,3*pi/4) q[13];
cx q[11],q[13];
u1(pi/4) q[13];
cx q[15],q[13];
u1(-pi/4) q[13];
cx q[11],q[13];
rx(pi/2) q[11];
u1(pi/4) q[13];
rzz(2.4330328969809467) q[14],q[11];
rx(-pi/2) q[11];
u2(pi/2,0) q[14];
cx q[14],q[1];
ry(-1.1062130204981022) q[1];
ry(1.1062130204981022) q[14];
cx q[14],q[1];
u1(-2.827970067271554) q[1];
cx q[1],q[0];
h q[0];
cu1(pi/2) q[1],q[0];
u2(pi/2,-pi/2) q[0];
u2(-pi/2,-pi) q[14];
cx q[15],q[13];
u2(pi/4,3*pi/4) q[13];
u2(0,-pi/2) q[15];
cx q[15],q[18];
cx q[18],q[15];
h q[18];
cx q[2],q[13];
u3(pi,1.502674433779375,-2.4243163832078665) q[13];
cx q[13],q[20];
ry(1.8654869871546247) q[13];
cx q[2],q[12];
tdg q[12];
ry(-1.8654869871546247) q[20];
cx q[13],q[20];
u3(3*pi/4,-pi,pi/2) q[13];
u1(-0.25222824412474054) q[20];

// rxx gate
h q[20];
h q[7];
cx q[20],q[7];
rz(2.5958092891759503) q[7];
cx q[20],q[7];
h q[7];
h q[20];

cswap q[3],q[9],q[21];
sx q[21];
cx q[17],q[21];
x q[17];
x q[3];
cu3(5.314723798560928,5.268488684001932,2.1504242006182763) q[4],q[18];
rz(6.219213711951733) q[4];
cz q[19],q[4];
cp(4.721333576446761) q[5],q[17];
cu1(pi/2) q[10],q[5];
u3(2.800722049322289,-pi/2,-pi) q[10];
cu1(1.184153166433536) q[6],q[9];
ccx q[21],q[3],q[9];
h q[3];
rx(4.4717645800568135) q[6];
cu3(0.5357030120304237,2.440535605737203,2.7482500772391254) q[20],q[6];
u2(pi/4,-pi) q[7];
cx q[18],q[7];
u2(0,3*pi/4) q[7];
cx q[17],q[7];
u1(pi/4) q[7];
cx q[8],q[12];
t q[12];
cx q[2],q[12];
u3(1.9128224785169774,1.7997458755134623,-2.4351182463566294) q[12];
ccx q[0],q[12],q[4];
u3(2.469727131739249,-pi,-pi) q[0];
cz q[15],q[2];
cx q[15],q[7];
cx q[2],q[3];
h q[3];
cu1(pi/2) q[2],q[3];
u1(-0.823036163138493) q[3];
u1(3.8235943477469303) q[4];
u1(-pi/4) q[7];
cx q[17],q[7];
u3(0.503351322995248,-0.42048212729976786,1.5278378787500628) q[17];
u1(pi/4) q[7];
cx q[15],q[7];
rzz(0.4780718686302382) q[15],q[1];
u2(-pi/2,1.5299985069170674) q[15];
cu1(pi/2) q[3],q[1];
cu1(0.9186755361331901) q[3],q[1];
u2(pi/4,3*pi/4) q[7];
cx q[18],q[7];
swap q[18],q[6];
u1(2.7599312071910997) q[18];
cu1(pi/2) q[3],q[18];
u1(-pi) q[18];
u1(-pi/4) q[7];
cx q[20],q[7];
rz(2.5003975968937953) q[7];
cx q[20],q[7];
z q[20];
cy q[8],q[11];
cu1(3.0400448093223384) q[11],q[21];
rx(pi/2) q[11];
rzz(1.331682425112953) q[11],q[13];
rx(-pi/2) q[11];
cx q[11],q[19];
cu1(2.6759532811131526) q[11],q[12];
u2(1.2723002500870975,2.9264084106706116) q[12];
rx(-pi/2) q[13];
ccx q[17],q[20],q[19];
cz q[5],q[21];
rx(pi/2) q[21];
cx q[5],q[7];
rz(0.3943431631506873) q[7];
cx q[5],q[7];
u2(0.12574175578903413,-pi) q[7];
u3(2.0700346564172443,-1.249026442621155,1.7262584392699765) q[8];
rzz(2.565716737441561) q[21],q[8];
rx(-pi/2) q[21];
u2(0,-pi/2) q[8];
cswap q[8],q[1],q[4];
cx q[1],q[3];
cx q[3],q[1];
cu1(6.164080651066891) q[18],q[3];
u2(0,0) q[18];
u1(1.5194019929882545) q[3];
rx(4.98365199249263) q[4];
u2(0,0) q[9];
cx q[9],q[16];
ry(-1.1796723193734187) q[16];
ry(-1.1796723193734187) q[9];
cx q[9],q[16];
u2(pi/4,-0.1964906865370306) q[16];
cx q[2],q[16];
u2(0,3*pi/4) q[16];
u2(-pi,-pi) q[9];
cx q[9],q[16];
u1(pi/4) q[16];
cx q[14],q[16];
u1(-pi/4) q[16];
cx q[9],q[16];
u1(pi/4) q[16];
cx q[14],q[16];
sx q[14];
cry(0.35036204201228127) q[14],q[21];
h q[14];
u2(pi/4,3*pi/4) q[16];
ccx q[17],q[19],q[14];
h q[14];
cry(3.1704170868245622) q[14],q[19];
u2(pi/4,-pi) q[19];
cx q[2],q[16];
u2(0,3*pi/4) q[16];
rzz(3.5897527146321706) q[16],q[6];
cx q[16],q[5];
u1(4.862323764687579) q[16];
ch q[16],q[8];
cry(5.4209241862562445) q[2],q[13];
ch q[2],q[13];
h q[13];
U(2.1884883783102658,4.9270092916743105,2.9201654642896124) q[21];u1(-2.9850325560467166) q[5];
cx q[15],q[5];
ry(-1.0274366467730753) q[15];
ry(-1.0274366467730753) q[5];
cx q[15],q[5];
u2(-pi,-pi) q[15];
u1(2.9850325560467166) q[5];
cu1(2.375127556967226) q[5],q[12];
cx q[12],q[19];
tdg q[19];
cz q[5],q[16];
p(5.4624388992719775) q[16];
rx(4.077226581120373) q[6];
cry(1.778294682141754) q[6],q[2];
h q[2];
cu1(pi/2) q[21],q[6];
h q[21];
ccx q[4],q[14],q[21];
cu1(pi/8) q[14],q[1];
h q[21];
y q[4];
u1(-pi/4) q[6];
rzz(5.385892556761493) q[6],q[21];
ccx q[7],q[11],q[13];
u2(-1.6792134263721463,1.1645061371573728) q[11];
h q[13];
cx q[13],q[2];
h q[2];
cu1(pi/2) q[13],q[2];
t q[13];
cu3(3.7102311819892546,5.536200438816542,0.7594781504422635) q[16],q[13];
h q[16];
u3(1.6379164194894182,0.9969935235381584,-1.1009729718733094) q[2];
u3(pi,2.896715891098596,-2.8967158910985966) q[7];
u3(pi,-3*pi/4,3*pi/4) q[8];
u2(0,-pi/2) q[9];
cx q[9],q[10];
cx q[10],q[9];
h q[10];
h q[9];
cx q[10],q[9];
rz(2.3661455902582555) q[9];
cx q[10],q[9];
cu1(pi/2) q[10],q[0];
h q[10];
cx q[17],q[10];
h q[10];
cu1(pi/2) q[17],q[10];
cx q[10],q[19];
t q[19];
cx q[12],q[19];
rx(pi/2) q[12];
u2(0,3*pi/4) q[19];
cx q[14],q[19];
cu1(-pi/8) q[19],q[1];
cx q[14],q[19];
cu1(pi/8) q[19],q[1];
cx q[19],q[10];
cu1(-pi/8) q[10],q[1];
cx q[14],q[10];
cu1(pi/8) q[10],q[1];
cx q[19],q[10];
cu1(-pi/8) q[10],q[1];
cx q[14],q[10];
cu1(pi/8) q[10],q[1];
x q[10];
u3(pi,2.2938987548290086,-2.2938987548290086) q[14];
h q[9];
crx(4.1000151280847605) q[20],q[9];
crz(2.5053932973696478) q[20],q[0];
u2(-pi/2,1.3872882630784495) q[0];
rzz(2.929122341901215) q[12],q[0];
rx(-pi/2) q[0];
cu1(pi/8) q[0],q[16];
rx(-pi/2) q[12];
crx(3.5176418043934525) q[12],q[6];
u3(0.9859643358688135,3.921952245886748,0.08722977146087585) q[6];
cp(0.10879080957487978) q[9],q[15];
cswap q[20],q[15],q[17];
sx q[17];
x q[20];
ch q[20],q[7];
cry(0.9702154593090756) q[4],q[15];
h q[15];
cx q[8],q[17];
u2(0,0) q[17];
cx q[17],q[3];
ry(2.0367627204859877) q[17];
ry(-2.0367627204859877) q[3];
cx q[17],q[3];
cx q[17],q[2];
ry(2.8118010620994327) q[17];
ry(-2.8118010620994327) q[2];
cx q[17],q[2];
u2(-pi,0) q[17];
u1(-0.1247158581970913) q[2];
u1(1.6221906606015386) q[3];
u3(0.3346984044049105,-2.745102552532198,0.3964901010575952) q[8];
cry(5.716371985477571) q[8],q[7];
p(3.3338991099112785) q[7];
u2(0,0) q[9];
cx q[9],q[11];
ry(-0.9739832503665405) q[11];
ry(-0.9739832503665405) q[9];
cx q[9],q[11];
u1(-2.8424635778221705) q[11];
cu3(2.4412882752101286,5.324549081630067,3.7619491323187204) q[11],q[5];
cx q[11],q[15];
rz(2.714580081289478) q[15];
cx q[11],q[15];
sxdg q[11];
h q[15];
cu1(pi/2) q[5],q[1];
u3(1.840696728733682,pi/2,-pi/2) q[1];
u2(-1.5492361889036994,-pi) q[9];
cx q[18],q[9];
ry(3.0149458932775706) q[18];
ry(-3.0149458932775706) q[9];
cx q[18],q[9];
u2(-pi,-pi) q[18];
cswap q[18],q[13],q[19];
u3(2.2459979918481423,-0.1556723411946379,-0.3718806383807589) q[13];
sx q[18];
u1(-1.5923564646860946) q[9];
cx q[0],q[9];
cu1(-pi/8) q[9],q[16];
cx q[0],q[9];
cu1(pi/8) q[9],q[16];
cx q[9],q[21];
cu1(-pi/8) q[21],q[16];
cx q[0],q[21];
cu1(pi/8) q[21],q[16];
cx q[9],q[21];
cu1(-pi/8) q[21],q[16];
cx q[0],q[21];
ch q[0],q[20];
sdg q[0];
cy q[20],q[14];
u2(pi/4,-pi/2) q[14];
ch q[20],q[13];
rx(5.620902208377852) q[20];
cu1(pi/8) q[21],q[16];
h q[16];
cu3(4.580422845973629,1.5990851979517735,0.6905962639033414) q[21],q[12];
u3(0.5297464685065629,-2.974992265625917,2.6151721707695916) q[12];
p(0.301053837539991) q[21];
cswap q[4],q[3],q[16];
swap q[16],q[2];
h q[2];
h q[4];
ccx q[18],q[15],q[4];
cx q[17],q[15];
cx q[15],q[17];
sx q[15];
ccx q[21],q[18],q[16];
u2(0,0) q[16];
h q[4];
u2(pi/4,-pi) q[9];
cx q[5],q[9];
u2(0,3*pi/4) q[9];
cx q[19],q[9];
u1(pi/4) q[9];
cx q[10],q[9];
u1(-pi/4) q[9];
cx q[19],q[9];
u1(pi/4) q[9];
cx q[10],q[9];
rz(1.8062675389819753) q[10];
ccx q[4],q[10],q[1];
h q[1];
swap q[21],q[1];
u2(0,0) q[1];
u3(0.3004053880842711,-0.733326637402981,-0.34446812023215667) q[21];
u3(1.1338229855330204,0,-pi/2) q[4];
u2(pi/4,3*pi/4) q[9];
cx q[5],q[9];
u2(pi/4,-pi) q[5];
cx q[19],q[5];
u2(0,3*pi/4) q[5];
cx q[3],q[5];
u1(pi/4) q[5];
cx q[8],q[5];
u1(-pi/4) q[5];
cx q[3],q[5];
u1(0.8271296853845325) q[3];
cx q[16],q[3];
ry(1.9798255533625133) q[16];
ry(-1.9798255533625133) q[3];
cx q[16],q[3];
u3(1.1252527303498907,-1.5710969656698137,0.2150527132235842) q[16];
u1(-0.041731521987085074) q[3];
cx q[4],q[3];
cx q[3],q[4];
h q[3];
cry(2.1076832721630585) q[16],q[3];
u3(0.6985636519815531,-pi,0) q[16];
u1(pi/4) q[5];
cx q[8],q[5];
u2(pi/4,3*pi/4) q[5];
cx q[19],q[5];
h q[19];
cu1(pi/8) q[11],q[19];
cx q[11],q[0];
cu1(-pi/8) q[0],q[19];
cx q[11],q[0];
cu1(pi/8) q[0],q[19];
cx q[0],q[7];
u3(1.976436916784817,1.057489168535362,2.2226772264465673) q[5];
cx q[5],q[14];
tdg q[14];
cx q[20],q[14];
t q[14];
h q[20];
cx q[5],q[14];
u2(0,3*pi/4) q[14];
cu1(-pi/8) q[7],q[19];
cx q[11],q[7];
cu1(pi/8) q[7],q[19];
cx q[0],q[7];
sdg q[0];
cu1(-pi/8) q[7],q[19];
cx q[11],q[7];
y q[11];
cp(2.175699586821472) q[11],q[0];
cu1(pi/8) q[7],q[19];
h q[19];
rzz(1.4243841952904288) q[17],q[19];
cx q[17],q[15];
cx q[15],q[17];
h q[15];
cu1(pi/2) q[11],q[15];
u1(2.4373642661831223) q[11];
h q[15];
h q[17];
cx q[0],q[17];
h q[17];
cu1(pi/2) q[0],q[17];
cp(3.07838862340904) q[7],q[18];
u1(2.2897896846086163) q[18];
u2(-pi/2,-pi/2) q[8];
u2(0,3*pi/4) q[9];
ch q[9],q[6];
rz(2.2630080158115735) q[6];
cry(1.1216013695609326) q[6],q[10];
rzz(2.374451738446438) q[10],q[8];
sdg q[10];
cp(3.2642050365801545) q[7],q[6];
h q[6];
cx q[5],q[6];
rz(3.6863167904415386) q[6];
cx q[5],q[6];
cswap q[10],q[0],q[5];
u3(2.410518314490394,3.449531222870809,2.1852899551563714) q[0];
cx q[21],q[0];
h q[0];
sxdg q[21];
h q[6];
z q[8];
cx q[9],q[2];
rz(3.2889385832906615) q[2];
cx q[9],q[2];
swap q[13],q[9];
u1(-1.692455558663964) q[13];
cx q[1],q[13];
ry(2.486695017227554) q[1];
ry(-2.486695017227554) q[13];
cx q[1],q[13];
u2(-pi,-pi) q[1];
cu1(pi/2) q[1],q[14];
u1(1.6924555586639638) q[13];
cu1(pi/8) q[13],q[20];
cx q[13],q[18];
cu1(-pi/8) q[18],q[20];
cx q[13],q[18];
cu1(pi/8) q[18],q[20];
u3(1.7404325703839565,3.0175842794389025,-pi/2) q[2];
cx q[18],q[2];
cu1(-pi/8) q[2],q[20];
cx q[13],q[2];
cu1(pi/8) q[2],q[20];
cx q[18],q[2];
cu1(-pi/8) q[2],q[20];
cx q[13],q[2];
cu1(pi/2) q[1],q[13];
u1(pi/2) q[1];
cu1(pi/2) q[11],q[1];
u1(-0.8687266409748426) q[1];
cu1(pi/8) q[2],q[20];

// rxx gate
h q[18];
h q[2];
cx q[18],q[2];
rz(6.26704953729603) q[2];
cx q[18],q[2];
h q[2];
h q[18];

cry(5.536305699572632) q[18],q[5];
u2(1.4984070887625043,-pi) q[20];
cu1(pi/2) q[9],q[19];
cu3(0.6783520474479109,1.1583140499519742,5.8969176696084515) q[19],q[4];
u2(pi/4,-pi) q[19];
cx q[14],q[19];
tdg q[19];
swap q[4],q[3];
ccx q[3],q[12],q[13];
u2(-pi/2,-pi) q[12];
cu1(pi/2) q[11],q[12];
u1(5.836980980008532) q[11];
u3(1.5300457827555594,pi/2,0.52578455560313) q[3];
cx q[3],q[1];
ry(-1.0128749997324702) q[1];
ry(-1.0128749997324702) q[3];
cx q[3],q[1];
u1(-0.2368817477477787) q[1];
u2(-pi,-pi) q[3];
cx q[6],q[19];
t q[19];
cx q[14],q[19];
cz q[14],q[10];
rx(pi/2) q[14];
rzz(5.3998709743359) q[14],q[16];
rx(-pi/2) q[14];
rx(-pi/2) q[16];
u2(0,3*pi/4) q[19];
u2(-pi/2,-pi) q[6];
cx q[18],q[6];
rz(4.142192030122678) q[6];
cx q[18],q[6];
u3(2.3596326474184526,0.6168033953421324,-2.187599722137029) q[18];
u1(pi/4) q[6];
cu3(5.0782433700995036,5.363650200973858,4.908630437566993) q[7],q[9];
ch q[8],q[7];
u3(pi,-0.13463800149709781,0.13463800149709781) q[7];
crx(2.8276519920871297) q[8],q[19];
cswap q[8],q[5],q[19];
cx q[19],q[6];
tdg q[6];
cx q[7],q[6];
t q[6];
cx q[19],q[6];
u2(-pi,-pi/4) q[6];
h q[8];
u2(pi/4,-pi) q[9];
cx q[15],q[9];
tdg q[9];
cx q[17],q[9];
z q[17];
t q[9];
cx q[15],q[9];
ccx q[15],q[4],q[2];
rzz(2.6330369831560847) q[10],q[15];
sx q[10];
rzz(0.07424098447119444) q[10],q[21];
cu1(pi/8) q[15],q[8];
cswap q[2],q[17],q[20];
cu1(5.2448052187979055) q[14],q[20];
u1(0.5773570584341212) q[14];
u1(-3*pi/4) q[17];
swap q[2],q[5];
z q[2];
u3(5.1902564084840614,4.359151593559829,-4.359151593559829) q[4];
cx q[4],q[0];
h q[0];
cu1(pi/2) q[4],q[0];
u3(3.0681767772205215,0.598056484472393,-1.6409957414706624) q[0];
cz q[20],q[4];
u1(-pi/4) q[9];
cx q[13],q[9];
u1(pi/2) q[13];
cx q[15],q[9];
cu1(-pi/8) q[9],q[8];
cx q[15],q[9];
cu1(pi/8) q[9],q[8];
cx q[9],q[16];
cu1(-pi/8) q[16],q[8];
cx q[15],q[16];
cu1(pi/8) q[16],q[8];
cx q[9],q[16];
cu1(-pi/8) q[16],q[8];
cx q[15],q[16];
cp(5.723088953577587) q[15],q[7];
cu1(pi/8) q[16],q[8];
rx(pi/2) q[16];
h q[8];
cy q[8],q[19];
rx(pi/2) q[9];
rzz(1.9404055173111059) q[16],q[9];
rx(-pi/2) q[16];
rx(-pi/2) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];
meas[5] = measure q[5];
meas[6] = measure q[6];
meas[7] = measure q[7];
meas[8] = measure q[8];
meas[9] = measure q[9];
meas[10] = measure q[10];
meas[11] = measure q[11];
meas[12] = measure q[12];
meas[13] = measure q[13];
meas[14] = measure q[14];
meas[15] = measure q[15];
meas[16] = measure q[16];
meas[17] = measure q[17];
meas[18] = measure q[18];
meas[19] = measure q[19];
meas[20] = measure q[20];
meas[21] = measure q[21];