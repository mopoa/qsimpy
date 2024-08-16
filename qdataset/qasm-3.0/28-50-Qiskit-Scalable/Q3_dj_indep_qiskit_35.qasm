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

qubit[35] q;
bit[34] c;
u2(0,0) q[0];
u2(0,0) q[1];
h q[2];
u2(0,0) q[3];
h q[4];
u2(0,0) q[5];
u2(0,0) q[6];
h q[7];
u2(0,0) q[8];
u2(0,0) q[9];
h q[10];
u2(0,0) q[11];
u2(0,0) q[12];
h q[13];
h q[14];
u2(0,0) q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
u2(0,0) q[21];
h q[22];
h q[23];
u2(0,0) q[24];
u2(0,0) q[25];
h q[26];
h q[27];
u2(0,0) q[28];
h q[29];
h q[30];
u2(0,0) q[31];
h q[32];
h q[33];
u2(-pi,-pi) q[34];
cx q[0],q[34];
u2(-pi,-pi) q[0];
cx q[1],q[34];
u2(-pi,-pi) q[1];
cx q[2],q[34];
h q[2];
cx q[3],q[34];
u2(-pi,-pi) q[3];
cx q[4],q[34];
h q[4];
cx q[5],q[34];
u2(-pi,-pi) q[5];
cx q[6],q[34];
u2(-pi,-pi) q[6];
cx q[7],q[34];
h q[7];
cx q[8],q[34];
u2(-pi,-pi) q[8];
cx q[9],q[34];
cx q[10],q[34];
h q[10];
cx q[11],q[34];
u2(-pi,-pi) q[11];
cx q[12],q[34];
u2(-pi,-pi) q[12];
cx q[13],q[34];
h q[13];
cx q[14],q[34];
h q[14];
cx q[15],q[34];
u2(-pi,-pi) q[15];
cx q[16],q[34];
h q[16];
cx q[17],q[34];
h q[17];
cx q[18],q[34];
h q[18];
cx q[19],q[34];
h q[19];
cx q[20],q[34];
h q[20];
cx q[21],q[34];
u2(-pi,-pi) q[21];
cx q[22],q[34];
h q[22];
cx q[23],q[34];
h q[23];
cx q[24],q[34];
u2(-pi,-pi) q[24];
cx q[25],q[34];
u2(-pi,-pi) q[25];
cx q[26],q[34];
h q[26];
cx q[27],q[34];
h q[27];
cx q[28],q[34];
u2(-pi,-pi) q[28];
cx q[29],q[34];
h q[29];
cx q[30],q[34];
h q[30];
cx q[31],q[34];
u2(-pi,-pi) q[31];
cx q[32],q[34];
h q[32];
cx q[33],q[34];
h q[33];
u2(-pi,-pi) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26],q[27],q[28],q[29],q[30],q[31],q[32],q[33],q[34];
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
c[21] = measure q[21];
c[22] = measure q[22];
c[23] = measure q[23];
c[24] = measure q[24];
c[25] = measure q[25];
c[26] = measure q[26];
c[27] = measure q[27];
c[28] = measure q[28];
c[29] = measure q[29];
c[30] = measure q[30];
c[31] = measure q[31];
c[32] = measure q[32];
c[33] = measure q[33];