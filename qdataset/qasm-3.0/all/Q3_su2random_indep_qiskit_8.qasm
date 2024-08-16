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

qubit[8] q;
bit[8] meas;
u3(1.4368347742816614,-2.079037930015222,-pi) q[0];
u3(0.13038834331032634,0.5550554224571163,0) q[1];
cx q[0],q[1];
u3(2.3018560275705338,1.1646500873100205,-pi) q[2];
cx q[0],q[2];
cx q[1],q[2];
u3(1.5783117544539513,2.8487544111850127,-pi) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
u3(3.1322119352256292,0.02480768898038388,0) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
u3(1.4124389803026796,-3.064986403423018,0) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
u3(1.2444656817555666,-1.1773372206208812,0) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
u3(1.5046299106322627,0.70702212977712,-pi) q[7];
cx q[0],q[7];
u3(1.748262901313365,-0.36547294383070916,-pi) q[0];
cx q[1],q[7];
u3(1.8339114230470699,2.72699034602209,0) q[1];
cx q[0],q[1];
cx q[2],q[7];
u3(0.5166404252966228,0.7399517487893483,-pi) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[7];
u3(1.7933732440688748,0.08255001257991035,-pi) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[7];
u3(2.8742785055981956,0.94497336375301,-pi) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[7];
u3(0.8932807542109366,-2.5067461861055573,0) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[7];
u3(2.345769178126651,-1.223818747839898,0) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
u3(2.047478881918867,0.13601306986195905,-pi) q[7];
cx q[0],q[7];
u3(0.5739760098973872,0.2989926356969477,-pi) q[0];
cx q[1],q[7];
u3(2.0058195038543025,-1.1354532936221062,0) q[1];
cx q[0],q[1];
cx q[2],q[7];
u3(0.5683728542359916,1.2500242582094412,0) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[7];
u3(1.8893541777246623,-0.899436076324772,0) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[7];
u3(0.716184867709297,2.2094986973106154,0) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[7];
u3(1.0764269733890999,1.5999986339275987,-pi) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[7];
u3(0.2946582649473353,1.8595822481541884,0) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
u3(2.348107098834042,2.4123440472691007,-pi) q[7];
cx q[0],q[7];
u3(2.045249940143549,-0.0777514342798824,0) q[0];
cx q[1],q[7];
u3(1.0368254640000032,2.867046025905182,0) q[1];
cx q[2],q[7];
u3(2.4663339782035085,-1.0925023928214674,0) q[2];
cx q[3],q[7];
u3(0.5872288522304125,1.5794302666547848,0) q[3];
cx q[4],q[7];
u3(1.1240263022165689,0.6118041095001514,-pi) q[4];
cx q[5],q[7];
u3(0.9497161489686775,-0.6105260558088226,0) q[5];
cx q[6],q[7];
u3(2.4134622602982154,-2.9244586574319644,0) q[6];
u3(0.3502202738938201,0.5667518785975818,-pi) q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];
meas[5] = measure q[5];
meas[6] = measure q[6];
meas[7] = measure q[7];