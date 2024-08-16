// Generated from qBraid v0.7.1
OPENQASM 3.0;
include 'stdgates.inc';
// Generated from qiskit v0.24.0 file qiskit/qasm/libs/qelib1.inc

// QASM 3 equivalent definitions of qelib1.inc gates

// idle gate (identity) with length gamma*sqglen
 
// generic single qubit gate

// inverse sqrt(X)

// controlled-sqrt(X)

// controlled phase rotation 

// controlled-U

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

// relative-phase CCX

// relative-phase 3-controlled X gate

// 3-controlled X gate

// 3-controlled sqrt(X) gate, this equals the C3X gate 
// where the CU1 rotations are -pi/8 not -pi/4

// 4-controlled X gate 

qubit[9] q;
bit[9] meas;
u3(1.4368347742816616,-2.586537231132677,-pi) q[0];
u3(0.13038834331032634,-1.9769425662797726,0) q[1];
cx q[0],q[1];
u3(2.3018560275705338,2.8487544111850127,-pi) q[2];
cx q[0],q[2];
cx q[1],q[2];
u3(1.5783117544539513,-3.1167849646094092,-pi) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
u3(3.1322119352256292,-3.064986403423018,0) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
u3(1.4124389803026798,-1.1773372206208812,0) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
u3(1.2444656817555668,-2.434570523812673,0) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
u3(1.5046299106322627,1.3933297522764283,-pi) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
u3(1.0625547235745711,1.8339114230470699,0) q[8];
cx q[0],q[8];
u3(0.5166404252966229,0.0825500125799099,-pi) q[0];
cx q[1],q[8];
u3(1.7933732440688746,0.94497336375301,-pi) q[1];
cx q[0],q[1];
cx q[2],q[8];
u3(2.8742785055981956,0.6348464674842362,-pi) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[8];
u3(0.8932807542109366,-1.223818747839898,0) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[8];
u3(2.345769178126651,-3.005579583727833,0) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[8];
u3(2.047478881918867,2.567616643692406,-pi) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[8];
u3(2.7761197097590844,2.005819503854303,0) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
cx q[7],q[8];
u3(2.72699034602209,0.5683728542359914,0) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
u3(2.4016409048004452,-1.252238475865131,-pi) q[8];
cx q[0],q[8];
u3(0.716184867709297,-1.5415940196621944,0) q[0];
cx q[1],q[8];
u3(1.0764269733890999,-1.2820104054356043,-pi) q[1];
cx q[0],q[1];
cx q[2],q[8];
u3(0.29465826494733527,-0.7292486063206916,0) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[8];
u3(2.348107098834042,-1.0963427134462451,-pi) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[8];
u3(2.842600017892846,-2.10476718958979,-pi) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[8];
u3(1.1354532936221056,-0.6752586753862846,-pi) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[8];
u3(1.2500242582094414,0.5872288522304121,0) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
cx q[7],q[8];
u3(0.8994360763247731,2.0175663513732243,-pi) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
u3(2.2094986973106154,0.9497161489686778,0) q[8];
cx q[0],q[8];
u3(2.4134622602982154,-2.574840774992211,0) q[0];
cx q[1],q[8];
u3(0.3502202738938201,-2.894778030919191,-pi) q[1];
cx q[2],q[8];
u3(0.07775143427988175,-0.8973534757447101,-pi) q[2];
cx q[3],q[8];
u3(2.8670460259051826,0.5002237983271178,0) q[3];
cx q[4],q[8];
u3(1.0925023928214677,-1.2223313827259215,-pi) q[4];
cx q[5],q[8];
u3(1.5794302666547846,2.0779707218466736,0) q[5];
cx q[6],q[8];
u3(2.5297885440896417,1.7205264938110494,-pi) q[6];
cx q[7],q[8];
u3(0.6105260558088238,-2.890521540662405,-pi) q[7];
u3(2.924458657431965,-0.4430137085195698,-pi) q[8];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];
meas[5] = measure q[5];
meas[6] = measure q[6];
meas[7] = measure q[7];
meas[8] = measure q[8];