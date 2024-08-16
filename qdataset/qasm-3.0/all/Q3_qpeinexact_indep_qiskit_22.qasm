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

qubit[21] q;

qubit[1] psi;
bit[21] c;
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
x psi[0];
cp(0.2047370002549985) psi[0],q[0];
cp(0.409474000509997) psi[0],q[1];
cp(0.818948001019994) psi[0],q[2];
cp(1.637896002039988) psi[0],q[3];
cp(-3.0073933030996103) psi[0],q[4];
cp(0.2683987009803658) psi[0],q[5];
cp(0.5367974019607316) psi[0],q[6];
cp(1.0735948039214631) psi[0],q[7];
cp(2.1471896078429262) psi[0],q[8];
cp(-1.9888060914937338) psi[0],q[9];
cp(2.3055731241921187) psi[0],q[10];
cp(-1.6720390587953489) psi[0],q[11];
cp(2.9391071895888885) psi[0],q[12];
cp(-0.4049709280018093) psi[0],q[13];
cp(-0.8099418560036186) psi[0],q[14];
cp(-1.6198837120072371) psi[0],q[15];
cp(3.043417883165112) psi[0],q[16];
cp(-pi/16) psi[0],q[17];
cp(-pi/8) psi[0],q[18];
cp(-pi/4) psi[0],q[19];
cp(-pi/2) psi[0],q[20];
swap q[0],q[20];
h q[0];
swap q[1],q[19];
cp(-pi/2) q[1],q[0];
h q[1];
swap q[2],q[18];
cp(-pi/4) q[2],q[0];
cp(-pi/2) q[2],q[1];
h q[2];
swap q[3],q[17];
cp(-pi/8) q[3],q[0];
cp(-pi/4) q[3],q[1];
cp(-pi/2) q[3],q[2];
h q[3];
swap q[4],q[16];
cp(-pi/16) q[4],q[0];
cp(-pi/8) q[4],q[1];
cp(-pi/4) q[4],q[2];
cp(-pi/2) q[4],q[3];
h q[4];
swap q[5],q[15];
cp(-pi/32) q[5],q[0];
cp(-pi/16) q[5],q[1];
cp(-pi/8) q[5],q[2];
cp(-pi/4) q[5],q[3];
cp(-pi/2) q[5],q[4];
h q[5];
swap q[6],q[14];
cp(-pi/64) q[6],q[0];
cp(-pi/32) q[6],q[1];
cp(-pi/16) q[6],q[2];
cp(-pi/8) q[6],q[3];
cp(-pi/4) q[6],q[4];
cp(-pi/2) q[6],q[5];
h q[6];
swap q[7],q[13];
cp(-pi/128) q[7],q[0];
cp(-pi/64) q[7],q[1];
cp(-pi/32) q[7],q[2];
cp(-pi/16) q[7],q[3];
cp(-pi/8) q[7],q[4];
cp(-pi/4) q[7],q[5];
cp(-pi/2) q[7],q[6];
h q[7];
swap q[8],q[12];
cp(-pi/256) q[8],q[0];
cp(-pi/128) q[8],q[1];
cp(-pi/64) q[8],q[2];
cp(-pi/32) q[8],q[3];
cp(-pi/16) q[8],q[4];
cp(-pi/8) q[8],q[5];
cp(-pi/4) q[8],q[6];
cp(-pi/2) q[8],q[7];
h q[8];
swap q[9],q[11];
cp(-pi/512) q[9],q[0];
cp(-pi/1024) q[10],q[0];
cp(-pi/2048) q[11],q[0];
cp(-pi/4096) q[12],q[0];
cp(-pi/8192) q[13],q[0];
cp(-pi/16384) q[14],q[0];
cp(-pi/32768) q[15],q[0];
cp(-pi/65536) q[16],q[0];
cp(-pi/131072) q[17],q[0];
cp(-pi/262144) q[18],q[0];
cp(-pi/524288) q[19],q[0];
cp(-pi/1048576) q[20],q[0];
cp(-pi/256) q[9],q[1];
cp(-pi/512) q[10],q[1];
cp(-pi/1024) q[11],q[1];
cp(-pi/2048) q[12],q[1];
cp(-pi/4096) q[13],q[1];
cp(-pi/8192) q[14],q[1];
cp(-pi/16384) q[15],q[1];
cp(-pi/32768) q[16],q[1];
cp(-pi/65536) q[17],q[1];
cp(-pi/131072) q[18],q[1];
cp(-pi/262144) q[19],q[1];
cp(-pi/524288) q[20],q[1];
cp(-pi/128) q[9],q[2];
cp(-pi/256) q[10],q[2];
cp(-pi/512) q[11],q[2];
cp(-pi/1024) q[12],q[2];
cp(-pi/2048) q[13],q[2];
cp(-pi/4096) q[14],q[2];
cp(-pi/8192) q[15],q[2];
cp(-pi/16384) q[16],q[2];
cp(-pi/32768) q[17],q[2];
cp(-pi/65536) q[18],q[2];
cp(-pi/131072) q[19],q[2];
cp(-pi/262144) q[20],q[2];
cp(-pi/64) q[9],q[3];
cp(-pi/128) q[10],q[3];
cp(-pi/256) q[11],q[3];
cp(-pi/512) q[12],q[3];
cp(-pi/1024) q[13],q[3];
cp(-pi/2048) q[14],q[3];
cp(-pi/4096) q[15],q[3];
cp(-pi/8192) q[16],q[3];
cp(-pi/16384) q[17],q[3];
cp(-pi/32768) q[18],q[3];
cp(-pi/65536) q[19],q[3];
cp(-pi/131072) q[20],q[3];
cp(-pi/32) q[9],q[4];
cp(-pi/64) q[10],q[4];
cp(-pi/128) q[11],q[4];
cp(-pi/256) q[12],q[4];
cp(-pi/512) q[13],q[4];
cp(-pi/1024) q[14],q[4];
cp(-pi/2048) q[15],q[4];
cp(-pi/4096) q[16],q[4];
cp(-pi/8192) q[17],q[4];
cp(-pi/16384) q[18],q[4];
cp(-pi/32768) q[19],q[4];
cp(-pi/65536) q[20],q[4];
cp(-pi/16) q[9],q[5];
cp(-pi/32) q[10],q[5];
cp(-pi/64) q[11],q[5];
cp(-pi/128) q[12],q[5];
cp(-pi/256) q[13],q[5];
cp(-pi/512) q[14],q[5];
cp(-pi/1024) q[15],q[5];
cp(-pi/2048) q[16],q[5];
cp(-pi/4096) q[17],q[5];
cp(-pi/8192) q[18],q[5];
cp(-pi/16384) q[19],q[5];
cp(-pi/32768) q[20],q[5];
cp(-pi/8) q[9],q[6];
cp(-pi/16) q[10],q[6];
cp(-pi/32) q[11],q[6];
cp(-pi/64) q[12],q[6];
cp(-pi/128) q[13],q[6];
cp(-pi/256) q[14],q[6];
cp(-pi/512) q[15],q[6];
cp(-pi/1024) q[16],q[6];
cp(-pi/2048) q[17],q[6];
cp(-pi/4096) q[18],q[6];
cp(-pi/8192) q[19],q[6];
cp(-pi/16384) q[20],q[6];
cp(-pi/4) q[9],q[7];
cp(-pi/8) q[10],q[7];
cp(-pi/16) q[11],q[7];
cp(-pi/32) q[12],q[7];
cp(-pi/64) q[13],q[7];
cp(-pi/128) q[14],q[7];
cp(-pi/256) q[15],q[7];
cp(-pi/512) q[16],q[7];
cp(-pi/1024) q[17],q[7];
cp(-pi/2048) q[18],q[7];
cp(-pi/4096) q[19],q[7];
cp(-pi/8192) q[20],q[7];
cp(-pi/2) q[9],q[8];
cp(-pi/4) q[10],q[8];
cp(-pi/8) q[11],q[8];
cp(-pi/16) q[12],q[8];
cp(-pi/32) q[13],q[8];
cp(-pi/64) q[14],q[8];
cp(-pi/128) q[15],q[8];
cp(-pi/256) q[16],q[8];
cp(-pi/512) q[17],q[8];
cp(-pi/1024) q[18],q[8];
cp(-pi/2048) q[19],q[8];
cp(-pi/4096) q[20],q[8];
h q[9];
cp(-pi/2) q[10],q[9];
h q[10];
cp(-pi/4) q[11],q[9];
cp(-pi/2) q[11],q[10];
h q[11];
cp(-pi/8) q[12],q[9];
cp(-pi/4) q[12],q[10];
cp(-pi/2) q[12],q[11];
h q[12];
cp(-pi/16) q[13],q[9];
cp(-pi/8) q[13],q[10];
cp(-pi/4) q[13],q[11];
cp(-pi/2) q[13],q[12];
h q[13];
cp(-pi/32) q[14],q[9];
cp(-pi/16) q[14],q[10];
cp(-pi/8) q[14],q[11];
cp(-pi/4) q[14],q[12];
cp(-pi/2) q[14],q[13];
h q[14];
cp(-pi/64) q[15],q[9];
cp(-pi/32) q[15],q[10];
cp(-pi/16) q[15],q[11];
cp(-pi/8) q[15],q[12];
cp(-pi/4) q[15],q[13];
cp(-pi/2) q[15],q[14];
h q[15];
cp(-pi/128) q[16],q[9];
cp(-pi/64) q[16],q[10];
cp(-pi/32) q[16],q[11];
cp(-pi/16) q[16],q[12];
cp(-pi/8) q[16],q[13];
cp(-pi/4) q[16],q[14];
cp(-pi/2) q[16],q[15];
h q[16];
cp(-pi/256) q[17],q[9];
cp(-pi/128) q[17],q[10];
cp(-pi/64) q[17],q[11];
cp(-pi/32) q[17],q[12];
cp(-pi/16) q[17],q[13];
cp(-pi/8) q[17],q[14];
cp(-pi/4) q[17],q[15];
cp(-pi/2) q[17],q[16];
h q[17];
cp(-pi/512) q[18],q[9];
cp(-pi/256) q[18],q[10];
cp(-pi/128) q[18],q[11];
cp(-pi/64) q[18],q[12];
cp(-pi/32) q[18],q[13];
cp(-pi/16) q[18],q[14];
cp(-pi/8) q[18],q[15];
cp(-pi/4) q[18],q[16];
cp(-pi/2) q[18],q[17];
h q[18];
cp(-pi/1024) q[19],q[9];
cp(-pi/512) q[19],q[10];
cp(-pi/256) q[19],q[11];
cp(-pi/128) q[19],q[12];
cp(-pi/64) q[19],q[13];
cp(-pi/32) q[19],q[14];
cp(-pi/16) q[19],q[15];
cp(-pi/8) q[19],q[16];
cp(-pi/4) q[19],q[17];
cp(-pi/2) q[19],q[18];
h q[19];
cp(-pi/2048) q[20],q[9];
cp(-pi/1024) q[20],q[10];
cp(-pi/512) q[20],q[11];
cp(-pi/256) q[20],q[12];
cp(-pi/128) q[20],q[13];
cp(-pi/64) q[20],q[14];
cp(-pi/32) q[20],q[15];
cp(-pi/16) q[20],q[16];
cp(-pi/8) q[20],q[17];
cp(-pi/4) q[20],q[18];
cp(-pi/2) q[20],q[19];
h q[20];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],psi[0];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
c[4] = measure q[4];
c[5] = measure q[5];
c[6] = measure q[6];
c[7] = measure q[7];
c[8] = measure q[8];
c[9] = measure q[9];
c[10] = measure q[10];
c[11] = measure q[11];
c[12] = measure q[12];
c[13] = measure q[13];
c[14] = measure q[14];
c[15] = measure q[15];
c[16] = measure q[16];
c[17] = measure q[17];
c[18] = measure q[18];
c[19] = measure q[19];
c[20] = measure q[20];