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

qubit[38] q;
bit[38] meas;
ry(4.84635053289793) q[0];
ry(0.130388343310326) q[1];
cx q[0],q[1];
ry(3.98132927960905) q[2];
cx q[0],q[2];
cx q[1],q[2];
ry(4.70487355272564) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
ry(3.13221193522563) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
ry(1.41243898030268) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
ry(1.24446568175557) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
ry(4.77855539654732) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
ry(1.06255472357457) q[8];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
ry(0.555055422457116) q[9];
cx q[0],q[9];
cx q[1],q[9];
cx q[2],q[9];
cx q[3],q[9];
cx q[4],q[9];
cx q[5],q[9];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
ry(4.30624274089981) q[10];
cx q[0],q[10];
cx q[1],q[10];
cx q[2],q[10];
cx q[3],q[10];
cx q[4],q[10];
cx q[5],q[10];
cx q[6],q[10];
cx q[7],q[10];
cx q[8],q[10];
cx q[9],q[10];
ry(5.99034706477481) q[11];
cx q[0],q[11];
cx q[1],q[11];
cx q[2],q[11];
cx q[3],q[11];
cx q[4],q[11];
cx q[5],q[11];
cx q[6],q[11];
cx q[7],q[11];
cx q[8],q[11];
cx q[9],q[11];
cx q[10],q[11];
ry(0.024807688980384) q[12];
cx q[0],q[12];
cx q[1],q[12];
cx q[2],q[12];
cx q[3],q[12];
cx q[4],q[12];
cx q[5],q[12];
cx q[6],q[12];
cx q[7],q[12];
cx q[8],q[12];
cx q[9],q[12];
cx q[10],q[12];
cx q[11],q[12];
ry(3.21819890375657) q[13];
cx q[0],q[13];
cx q[1],q[13];
cx q[2],q[13];
cx q[3],q[13];
cx q[4],q[13];
cx q[5],q[13];
cx q[6],q[13];
cx q[7],q[13];
cx q[8],q[13];
cx q[9],q[13];
cx q[10],q[13];
cx q[11],q[13];
cx q[12],q[13];
ry(5.10584808655871) q[14];
cx q[0],q[14];
cx q[1],q[14];
cx q[2],q[14];
cx q[3],q[14];
cx q[4],q[14];
cx q[5],q[14];
cx q[6],q[14];
cx q[7],q[14];
cx q[8],q[14];
cx q[9],q[14];
cx q[10],q[14];
cx q[11],q[14];
cx q[12],q[14];
cx q[13],q[14];
ry(3.84861478336691) q[15];
cx q[0],q[15];
cx q[1],q[15];
cx q[2],q[15];
cx q[3],q[15];
cx q[4],q[15];
cx q[5],q[15];
cx q[6],q[15];
cx q[7],q[15];
cx q[8],q[15];
cx q[9],q[15];
cx q[10],q[15];
cx q[11],q[15];
cx q[12],q[15];
cx q[13],q[15];
cx q[14],q[15];
ry(4.53492240586622) q[16];
cx q[0],q[16];
cx q[1],q[16];
cx q[2],q[16];
cx q[3],q[16];
cx q[4],q[16];
cx q[5],q[16];
cx q[6],q[16];
cx q[7],q[16];
cx q[8],q[16];
cx q[9],q[16];
cx q[10],q[16];
cx q[11],q[16];
cx q[12],q[16];
cx q[13],q[16];
cx q[14],q[16];
cx q[15],q[16];
ry(1.83391142304707) q[17];
cx q[0],q[17];
cx q[1],q[17];
cx q[2],q[17];
cx q[3],q[17];
cx q[4],q[17];
cx q[5],q[17];
cx q[6],q[17];
cx q[7],q[17];
cx q[8],q[17];
cx q[9],q[17];
cx q[10],q[17];
cx q[11],q[17];
cx q[12],q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[15],q[17];
cx q[16],q[17];
ry(5.76654488188296) q[18];
cx q[0],q[18];
cx q[1],q[18];
cx q[2],q[18];
cx q[3],q[18];
cx q[4],q[18];
cx q[5],q[18];
cx q[6],q[18];
cx q[7],q[18];
cx q[8],q[18];
cx q[9],q[18];
cx q[10],q[18];
cx q[11],q[18];
cx q[12],q[18];
cx q[13],q[18];
cx q[14],q[18];
cx q[15],q[18];
cx q[16],q[18];
cx q[17],q[18];
ry(4.48981206311071) q[19];
cx q[0],q[19];
cx q[1],q[19];
cx q[2],q[19];
cx q[3],q[19];
cx q[4],q[19];
cx q[5],q[19];
cx q[6],q[19];
cx q[7],q[19];
cx q[8],q[19];
cx q[9],q[19];
cx q[10],q[19];
cx q[11],q[19];
cx q[12],q[19];
cx q[13],q[19];
cx q[14],q[19];
cx q[15],q[19];
cx q[16],q[19];
cx q[17],q[19];
cx q[18],q[19];
ry(3.40890680158139) q[20];
cx q[0],q[20];
cx q[1],q[20];
cx q[2],q[20];
cx q[3],q[20];
cx q[4],q[20];
cx q[5],q[20];
cx q[6],q[20];
cx q[7],q[20];
cx q[8],q[20];
cx q[9],q[20];
cx q[10],q[20];
cx q[11],q[20];
cx q[12],q[20];
cx q[13],q[20];
cx q[14],q[20];
cx q[15],q[20];
cx q[16],q[20];
cx q[17],q[20];
cx q[18],q[20];
cx q[19],q[20];
ry(0.893280754210937) q[21];
cx q[0],q[21];
cx q[1],q[21];
cx q[2],q[21];
cx q[3],q[21];
cx q[4],q[21];
cx q[5],q[21];
cx q[6],q[21];
cx q[7],q[21];
cx q[8],q[21];
cx q[9],q[21];
cx q[10],q[21];
cx q[11],q[21];
cx q[12],q[21];
cx q[13],q[21];
cx q[14],q[21];
cx q[15],q[21];
cx q[16],q[21];
cx q[17],q[21];
cx q[18],q[21];
cx q[19],q[21];
cx q[20],q[21];
ry(2.34576917812665) q[22];
cx q[0],q[22];
cx q[1],q[22];
cx q[2],q[22];
cx q[3],q[22];
cx q[4],q[22];
cx q[5],q[22];
cx q[6],q[22];
cx q[7],q[22];
cx q[8],q[22];
cx q[9],q[22];
cx q[10],q[22];
cx q[11],q[22];
cx q[12],q[22];
cx q[13],q[22];
cx q[14],q[22];
cx q[15],q[22];
cx q[16],q[22];
cx q[17],q[22];
cx q[18],q[22];
cx q[19],q[22];
cx q[20],q[22];
cx q[21],q[22];
ry(4.23570642526072) q[23];
cx q[0],q[23];
cx q[1],q[23];
cx q[2],q[23];
cx q[3],q[23];
cx q[4],q[23];
cx q[5],q[23];
cx q[6],q[23];
cx q[7],q[23];
cx q[8],q[23];
cx q[9],q[23];
cx q[10],q[23];
cx q[11],q[23];
cx q[12],q[23];
cx q[13],q[23];
cx q[14],q[23];
cx q[15],q[23];
cx q[16],q[23];
cx q[17],q[23];
cx q[18],q[23];
cx q[19],q[23];
cx q[20],q[23];
cx q[21],q[23];
cx q[22],q[23];
ry(2.77611970975908) q[24];
cx q[0],q[24];
cx q[1],q[24];
cx q[2],q[24];
cx q[3],q[24];
cx q[4],q[24];
cx q[5],q[24];
cx q[6],q[24];
cx q[7],q[24];
cx q[8],q[24];
cx q[9],q[24];
cx q[10],q[24];
cx q[11],q[24];
cx q[12],q[24];
cx q[13],q[24];
cx q[14],q[24];
cx q[15],q[24];
cx q[16],q[24];
cx q[17],q[24];
cx q[18],q[24];
cx q[19],q[24];
cx q[20],q[24];
cx q[21],q[24];
cx q[22],q[24];
cx q[23],q[24];
ry(2.72699034602209) q[25];
cx q[0],q[25];
cx q[1],q[25];
cx q[2],q[25];
cx q[3],q[25];
cx q[4],q[25];
cx q[5],q[25];
cx q[6],q[25];
cx q[7],q[25];
cx q[8],q[25];
cx q[9],q[25];
cx q[10],q[25];
cx q[11],q[25];
cx q[12],q[25];
cx q[13],q[25];
cx q[14],q[25];
cx q[15],q[25];
cx q[16],q[25];
cx q[17],q[25];
cx q[18],q[25];
cx q[19],q[25];
cx q[20],q[25];
cx q[21],q[25];
cx q[22],q[25];
cx q[23],q[25];
cx q[24],q[25];
ry(3.88154440237914) q[26];
cx q[0],q[26];
cx q[1],q[26];
cx q[2],q[26];
cx q[3],q[26];
cx q[4],q[26];
cx q[5],q[26];
cx q[6],q[26];
cx q[7],q[26];
cx q[8],q[26];
cx q[9],q[26];
cx q[10],q[26];
cx q[11],q[26];
cx q[12],q[26];
cx q[13],q[26];
cx q[14],q[26];
cx q[15],q[26];
cx q[16],q[26];
cx q[17],q[26];
cx q[18],q[26];
cx q[19],q[26];
cx q[20],q[26];
cx q[21],q[26];
cx q[22],q[26];
cx q[23],q[26];
cx q[24],q[26];
cx q[25],q[26];
ry(3.2241426661697) q[27];
cx q[0],q[27];
cx q[1],q[27];
cx q[2],q[27];
cx q[3],q[27];
cx q[4],q[27];
cx q[5],q[27];
cx q[6],q[27];
cx q[7],q[27];
cx q[8],q[27];
cx q[9],q[27];
cx q[10],q[27];
cx q[11],q[27];
cx q[12],q[27];
cx q[13],q[27];
cx q[14],q[27];
cx q[15],q[27];
cx q[16],q[27];
cx q[17],q[27];
cx q[18],q[27];
cx q[19],q[27];
cx q[20],q[27];
cx q[21],q[27];
cx q[22],q[27];
cx q[23],q[27];
cx q[24],q[27];
cx q[25],q[27];
cx q[26],q[27];
ry(4.0865660173428) q[28];
cx q[0],q[28];
cx q[1],q[28];
cx q[2],q[28];
cx q[3],q[28];
cx q[4],q[28];
cx q[5],q[28];
cx q[6],q[28];
cx q[7],q[28];
cx q[8],q[28];
cx q[9],q[28];
cx q[10],q[28];
cx q[11],q[28];
cx q[12],q[28];
cx q[13],q[28];
cx q[14],q[28];
cx q[15],q[28];
cx q[16],q[28];
cx q[17],q[28];
cx q[18],q[28];
cx q[19],q[28];
cx q[20],q[28];
cx q[21],q[28];
cx q[22],q[28];
cx q[23],q[28];
cx q[24],q[28];
cx q[25],q[28];
cx q[26],q[28];
cx q[27],q[28];
ry(3.77643912107403) q[29];
cx q[0],q[29];
cx q[1],q[29];
cx q[2],q[29];
cx q[3],q[29];
cx q[4],q[29];
cx q[5],q[29];
cx q[6],q[29];
cx q[7],q[29];
cx q[8],q[29];
cx q[9],q[29];
cx q[10],q[29];
cx q[11],q[29];
cx q[12],q[29];
cx q[13],q[29];
cx q[14],q[29];
cx q[15],q[29];
cx q[16],q[29];
cx q[17],q[29];
cx q[18],q[29];
cx q[19],q[29];
cx q[20],q[29];
cx q[21],q[29];
cx q[22],q[29];
cx q[23],q[29];
cx q[24],q[29];
cx q[25],q[29];
cx q[26],q[29];
cx q[27],q[29];
cx q[28],q[29];
ry(5.05936655933969) q[30];
cx q[0],q[30];
cx q[1],q[30];
cx q[2],q[30];
cx q[3],q[30];
cx q[4],q[30];
cx q[5],q[30];
cx q[6],q[30];
cx q[7],q[30];
cx q[8],q[30];
cx q[9],q[30];
cx q[10],q[30];
cx q[11],q[30];
cx q[12],q[30];
cx q[13],q[30];
cx q[14],q[30];
cx q[15],q[30];
cx q[16],q[30];
cx q[17],q[30];
cx q[18],q[30];
cx q[19],q[30];
cx q[20],q[30];
cx q[21],q[30];
cx q[22],q[30];
cx q[23],q[30];
cx q[24],q[30];
cx q[25],q[30];
cx q[26],q[30];
cx q[27],q[30];
cx q[28],q[30];
cx q[29],q[30];
ry(3.27760572345175) q[31];
cx q[0],q[31];
cx q[1],q[31];
cx q[2],q[31];
cx q[3],q[31];
cx q[4],q[31];
cx q[5],q[31];
cx q[6],q[31];
cx q[7],q[31];
cx q[8],q[31];
cx q[9],q[31];
cx q[10],q[31];
cx q[11],q[31];
cx q[12],q[31];
cx q[13],q[31];
cx q[14],q[31];
cx q[15],q[31];
cx q[16],q[31];
cx q[17],q[31];
cx q[18],q[31];
cx q[19],q[31];
cx q[20],q[31];
cx q[21],q[31];
cx q[22],q[31];
cx q[23],q[31];
cx q[24],q[31];
cx q[25],q[31];
cx q[26],q[31];
cx q[27],q[31];
cx q[28],q[31];
cx q[29],q[31];
cx q[30],q[31];
ry(5.7092092972822) q[32];
cx q[0],q[32];
cx q[1],q[32];
cx q[2],q[32];
cx q[3],q[32];
cx q[4],q[32];
cx q[5],q[32];
cx q[6],q[32];
cx q[7],q[32];
cx q[8],q[32];
cx q[9],q[32];
cx q[10],q[32];
cx q[11],q[32];
cx q[12],q[32];
cx q[13],q[32];
cx q[14],q[32];
cx q[15],q[32];
cx q[16],q[32];
cx q[17],q[32];
cx q[18],q[32];
cx q[19],q[32];
cx q[20],q[32];
cx q[21],q[32];
cx q[22],q[32];
cx q[23],q[32];
cx q[24],q[32];
cx q[25],q[32];
cx q[26],q[32];
cx q[27],q[32];
cx q[28],q[32];
cx q[29],q[32];
cx q[30],q[32];
cx q[31],q[32];
ry(2.0058195038543) q[33];
cx q[0],q[33];
cx q[1],q[33];
cx q[2],q[33];
cx q[3],q[33];
cx q[4],q[33];
cx q[5],q[33];
cx q[6],q[33];
cx q[7],q[33];
cx q[8],q[33];
cx q[9],q[33];
cx q[10],q[33];
cx q[11],q[33];
cx q[12],q[33];
cx q[13],q[33];
cx q[14],q[33];
cx q[15],q[33];
cx q[16],q[33];
cx q[17],q[33];
cx q[18],q[33];
cx q[19],q[33];
cx q[20],q[33];
cx q[21],q[33];
cx q[22],q[33];
cx q[23],q[33];
cx q[24],q[33];
cx q[25],q[33];
cx q[26],q[33];
cx q[27],q[33];
cx q[28],q[33];
cx q[29],q[33];
cx q[30],q[33];
cx q[31],q[33];
cx q[32],q[33];
ry(0.568372854235992) q[34];
cx q[0],q[34];
cx q[1],q[34];
cx q[2],q[34];
cx q[3],q[34];
cx q[4],q[34];
cx q[5],q[34];
cx q[6],q[34];
cx q[7],q[34];
cx q[8],q[34];
cx q[9],q[34];
cx q[10],q[34];
cx q[11],q[34];
cx q[12],q[34];
cx q[13],q[34];
cx q[14],q[34];
cx q[15],q[34];
cx q[16],q[34];
cx q[17],q[34];
cx q[18],q[34];
cx q[19],q[34];
cx q[20],q[34];
cx q[21],q[34];
cx q[22],q[34];
cx q[23],q[34];
cx q[24],q[34];
cx q[25],q[34];
cx q[26],q[34];
cx q[27],q[34];
cx q[28],q[34];
cx q[29],q[34];
cx q[30],q[34];
cx q[31],q[34];
cx q[32],q[34];
cx q[33],q[34];
ry(1.88935417772466) q[35];
cx q[0],q[35];
cx q[1],q[35];
cx q[2],q[35];
cx q[3],q[35];
cx q[4],q[35];
cx q[5],q[35];
cx q[6],q[35];
cx q[7],q[35];
cx q[8],q[35];
cx q[9],q[35];
cx q[10],q[35];
cx q[11],q[35];
cx q[12],q[35];
cx q[13],q[35];
cx q[14],q[35];
cx q[15],q[35];
cx q[16],q[35];
cx q[17],q[35];
cx q[18],q[35];
cx q[19],q[35];
cx q[20],q[35];
cx q[21],q[35];
cx q[22],q[35];
cx q[23],q[35];
cx q[24],q[35];
cx q[25],q[35];
cx q[26],q[35];
cx q[27],q[35];
cx q[28],q[35];
cx q[29],q[35];
cx q[30],q[35];
cx q[31],q[35];
cx q[32],q[35];
cx q[33],q[35];
cx q[34],q[35];
ry(0.716184867709297) q[36];
cx q[0],q[36];
cx q[1],q[36];
cx q[2],q[36];
cx q[3],q[36];
cx q[4],q[36];
cx q[5],q[36];
cx q[6],q[36];
cx q[7],q[36];
cx q[8],q[36];
cx q[9],q[36];
cx q[10],q[36];
cx q[11],q[36];
cx q[12],q[36];
cx q[13],q[36];
cx q[14],q[36];
cx q[15],q[36];
cx q[16],q[36];
cx q[17],q[36];
cx q[18],q[36];
cx q[19],q[36];
cx q[20],q[36];
cx q[21],q[36];
cx q[22],q[36];
cx q[23],q[36];
cx q[24],q[36];
cx q[25],q[36];
cx q[26],q[36];
cx q[27],q[36];
cx q[28],q[36];
cx q[29],q[36];
cx q[30],q[36];
cx q[31],q[36];
cx q[32],q[36];
cx q[33],q[36];
cx q[34],q[36];
cx q[35],q[36];
ry(5.20675833379049) q[37];
cx q[0],q[37];
ry(0.294658264947335) q[0];
cx q[1],q[37];
ry(3.93507820834554) q[1];
cx q[0],q[1];
cx q[2],q[37];
ry(3.44058528928674) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[37];
ry(5.14773201355748) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[37];
ry(1.25002425820944) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[37];
ry(5.38374923085481) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[37];
ry(2.20949869731062) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
cx q[7],q[37];
ry(4.74159128751739) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
cx q[8],q[37];
ry(1.85958224815419) q[8];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
cx q[9],q[37];
cx q[10],q[37];
ry(2.04524994014355) q[10];
cx q[11],q[37];
ry(1.036825464) q[11];
cx q[12],q[37];
ry(2.46633397820351) q[12];
cx q[13],q[37];
ry(0.587228852230412) q[13];
cx q[14],q[37];
ry(5.15915900496302) q[14];
cx q[15],q[37];
ry(0.949716148968678) q[15];
cx q[16],q[37];
ry(2.41346226029822) q[16];
cx q[17],q[37];
ry(5.93296503328577) q[17];
cx q[18],q[37];
ry(6.2054338728997) q[18];
cx q[19],q[37];
ry(2.86704602590518) q[19];
cx q[20],q[37];
ry(5.19068291435812) q[20];
cx q[21],q[37];
ry(1.57943026665478) q[21];
cx q[22],q[37];
ry(3.75339676308994) q[22];
cx q[23],q[37];
ry(5.67265925137076) q[23];
cx q[24],q[37];
ry(3.35872664974762) q[24];
cx q[25],q[37];
ry(3.70834453218737) q[25];
cx q[26],q[37];
ry(0.246814622670602) q[26];
cx q[27],q[37];
ry(2.24423917784508) q[27];
cx q[28],q[37];
ry(0.500223798327118) q[28];
cx q[29],q[37];
ry(1.91926127086387) q[29];
cx q[30],q[37];
ry(2.07797072184667) q[30];
cx q[31],q[37];
ry(4.86211914740084) q[31];
cx q[32],q[37];
ry(0.251071112927388) q[32];
cx q[33],q[37];
ry(2.69857894507022) q[33];
cx q[34],q[37];
ry(1.97874389399801) q[34];
cx q[35],q[37];
ry(3.99919179827194) q[35];
cx q[36],q[37];
ry(2.17616332456639) q[36];
ry(0.270788675285506) q[37];
ry(5.55393670085889) q[9];
cx q[0],q[9];
cx q[0],q[10];
cx q[0],q[11];
cx q[0],q[12];
cx q[0],q[13];
cx q[0],q[14];
cx q[0],q[15];
cx q[0],q[16];
cx q[0],q[17];
cx q[0],q[18];
cx q[0],q[19];
cx q[0],q[20];
cx q[0],q[21];
cx q[0],q[22];
cx q[0],q[23];
cx q[0],q[24];
cx q[0],q[25];
cx q[0],q[26];
cx q[0],q[27];
cx q[0],q[28];
cx q[0],q[29];
cx q[0],q[30];
cx q[0],q[31];
cx q[0],q[32];
cx q[0],q[33];
cx q[0],q[34];
cx q[0],q[35];
cx q[0],q[36];
cx q[0],q[37];
ry(5.52867009609533) q[0];
cx q[1],q[9];
cx q[1],q[10];
cx q[1],q[11];
cx q[1],q[12];
cx q[1],q[13];
cx q[1],q[14];
cx q[1],q[15];
cx q[1],q[16];
cx q[1],q[17];
cx q[1],q[18];
cx q[1],q[19];
cx q[1],q[20];
cx q[1],q[21];
cx q[1],q[22];
cx q[1],q[23];
cx q[1],q[24];
cx q[1],q[25];
cx q[1],q[26];
cx q[1],q[27];
cx q[1],q[28];
cx q[1],q[29];
cx q[1],q[30];
cx q[1],q[31];
cx q[1],q[32];
cx q[1],q[33];
cx q[1],q[34];
cx q[1],q[35];
cx q[1],q[36];
cx q[1],q[37];
ry(4.7955820429843) q[1];
cx q[0],q[1];
cx q[2],q[9];
cx q[2],q[10];
cx q[2],q[11];
cx q[2],q[12];
cx q[2],q[13];
cx q[2],q[14];
cx q[2],q[15];
cx q[2],q[16];
cx q[2],q[17];
cx q[2],q[18];
cx q[2],q[19];
cx q[2],q[20];
cx q[2],q[21];
cx q[2],q[22];
cx q[2],q[23];
cx q[2],q[24];
cx q[2],q[25];
cx q[2],q[26];
cx q[2],q[27];
cx q[2],q[28];
cx q[2],q[29];
cx q[2],q[30];
cx q[2],q[31];
cx q[2],q[32];
cx q[2],q[33];
cx q[2],q[34];
cx q[2],q[35];
cx q[2],q[36];
cx q[2],q[37];
ry(5.51724392385255) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[9];
cx q[3],q[10];
cx q[3],q[11];
cx q[3],q[12];
cx q[3],q[13];
cx q[3],q[14];
cx q[3],q[15];
cx q[3],q[16];
cx q[3],q[17];
cx q[3],q[18];
cx q[3],q[19];
cx q[3],q[20];
cx q[3],q[21];
cx q[3],q[22];
cx q[3],q[23];
cx q[3],q[24];
cx q[3],q[25];
cx q[3],q[26];
cx q[3],q[27];
cx q[3],q[28];
cx q[3],q[29];
cx q[3],q[30];
cx q[3],q[31];
cx q[3],q[32];
cx q[3],q[33];
cx q[3],q[34];
cx q[3],q[35];
cx q[3],q[36];
cx q[3],q[37];
ry(2.62328731818401) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[9];
cx q[4],q[10];
cx q[4],q[11];
cx q[4],q[12];
cx q[4],q[13];
cx q[4],q[14];
cx q[4],q[15];
cx q[4],q[16];
cx q[4],q[17];
cx q[4],q[18];
cx q[4],q[19];
cx q[4],q[20];
cx q[4],q[21];
cx q[4],q[22];
cx q[4],q[23];
cx q[4],q[24];
cx q[4],q[25];
cx q[4],q[26];
cx q[4],q[27];
cx q[4],q[28];
cx q[4],q[29];
cx q[4],q[30];
cx q[4],q[31];
cx q[4],q[32];
cx q[4],q[33];
cx q[4],q[34];
cx q[4],q[35];
cx q[4],q[36];
cx q[4],q[37];
ry(3.80495605495645) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[9];
cx q[5],q[10];
cx q[5],q[11];
cx q[5],q[12];
cx q[5],q[13];
cx q[5],q[14];
cx q[5],q[15];
cx q[5],q[16];
cx q[5],q[17];
cx q[5],q[18];
cx q[5],q[19];
cx q[5],q[20];
cx q[5],q[21];
cx q[5],q[22];
cx q[5],q[23];
cx q[5],q[24];
cx q[5],q[25];
cx q[5],q[26];
cx q[5],q[27];
cx q[5],q[28];
cx q[5],q[29];
cx q[5],q[30];
cx q[5],q[31];
cx q[5],q[32];
cx q[5],q[33];
cx q[5],q[34];
cx q[5],q[35];
cx q[5],q[36];
cx q[5],q[37];
ry(3.22620596905881) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[9];
cx q[6],q[10];
cx q[6],q[11];
cx q[6],q[12];
cx q[6],q[13];
cx q[6],q[14];
cx q[6],q[15];
cx q[6],q[16];
cx q[6],q[17];
cx q[6],q[18];
cx q[6],q[19];
cx q[6],q[20];
cx q[6],q[21];
cx q[6],q[22];
cx q[6],q[23];
cx q[6],q[24];
cx q[6],q[25];
cx q[6],q[26];
cx q[6],q[27];
cx q[6],q[28];
cx q[6],q[29];
cx q[6],q[30];
cx q[6],q[31];
cx q[6],q[32];
cx q[6],q[33];
cx q[6],q[34];
cx q[6],q[35];
cx q[6],q[36];
cx q[6],q[37];
ry(3.75631844257445) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
cx q[7],q[9];
cx q[7],q[10];
cx q[7],q[11];
cx q[7],q[12];
cx q[7],q[13];
cx q[7],q[14];
cx q[7],q[15];
cx q[7],q[16];
cx q[7],q[17];
cx q[7],q[18];
cx q[7],q[19];
cx q[7],q[20];
cx q[7],q[21];
cx q[7],q[22];
cx q[7],q[23];
cx q[7],q[24];
cx q[7],q[25];
cx q[7],q[26];
cx q[7],q[27];
cx q[7],q[28];
cx q[7],q[29];
cx q[7],q[30];
cx q[7],q[31];
cx q[7],q[32];
cx q[7],q[33];
cx q[7],q[34];
cx q[7],q[35];
cx q[7],q[36];
cx q[7],q[37];
ry(1.64754958933665) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
cx q[8],q[9];
cx q[8],q[10];
cx q[8],q[11];
cx q[8],q[12];
cx q[8],q[13];
cx q[8],q[14];
cx q[8],q[15];
cx q[8],q[16];
cx q[8],q[17];
cx q[8],q[18];
cx q[8],q[19];
cx q[8],q[20];
cx q[8],q[21];
cx q[8],q[22];
cx q[8],q[23];
cx q[8],q[24];
cx q[8],q[25];
cx q[8],q[26];
cx q[8],q[27];
cx q[8],q[28];
cx q[8],q[29];
cx q[8],q[30];
cx q[8],q[31];
cx q[8],q[32];
cx q[8],q[33];
cx q[8],q[34];
cx q[8],q[35];
cx q[8],q[36];
cx q[8],q[37];
ry(1.89043018768814) q[8];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
cx q[9],q[10];
cx q[9],q[11];
cx q[10],q[11];
cx q[9],q[12];
cx q[10],q[12];
cx q[11],q[12];
cx q[9],q[13];
cx q[10],q[13];
cx q[11],q[13];
cx q[12],q[13];
cx q[9],q[14];
cx q[10],q[14];
cx q[11],q[14];
cx q[12],q[14];
cx q[13],q[14];
cx q[9],q[15];
cx q[10],q[15];
cx q[11],q[15];
cx q[12],q[15];
cx q[13],q[15];
cx q[14],q[15];
cx q[9],q[16];
cx q[10],q[16];
cx q[11],q[16];
cx q[12],q[16];
cx q[13],q[16];
cx q[14],q[16];
cx q[15],q[16];
cx q[9],q[17];
cx q[10],q[17];
cx q[11],q[17];
cx q[12],q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[15],q[17];
cx q[16],q[17];
cx q[9],q[18];
cx q[10],q[18];
cx q[11],q[18];
cx q[12],q[18];
cx q[13],q[18];
cx q[14],q[18];
cx q[15],q[18];
cx q[16],q[18];
cx q[17],q[18];
cx q[9],q[19];
cx q[10],q[19];
cx q[11],q[19];
cx q[12],q[19];
cx q[13],q[19];
cx q[14],q[19];
cx q[15],q[19];
cx q[16],q[19];
cx q[17],q[19];
cx q[18],q[19];
cx q[9],q[20];
cx q[10],q[20];
cx q[11],q[20];
cx q[12],q[20];
cx q[13],q[20];
cx q[14],q[20];
cx q[15],q[20];
cx q[16],q[20];
cx q[17],q[20];
cx q[18],q[20];
cx q[19],q[20];
cx q[9],q[21];
cx q[10],q[21];
cx q[11],q[21];
cx q[12],q[21];
cx q[13],q[21];
cx q[14],q[21];
cx q[15],q[21];
cx q[16],q[21];
cx q[17],q[21];
cx q[18],q[21];
cx q[19],q[21];
cx q[20],q[21];
cx q[9],q[22];
cx q[10],q[22];
cx q[11],q[22];
cx q[12],q[22];
cx q[13],q[22];
cx q[14],q[22];
cx q[15],q[22];
cx q[16],q[22];
cx q[17],q[22];
cx q[18],q[22];
cx q[19],q[22];
cx q[20],q[22];
cx q[21],q[22];
cx q[9],q[23];
cx q[10],q[23];
cx q[11],q[23];
cx q[12],q[23];
cx q[13],q[23];
cx q[14],q[23];
cx q[15],q[23];
cx q[16],q[23];
cx q[17],q[23];
cx q[18],q[23];
cx q[19],q[23];
cx q[20],q[23];
cx q[21],q[23];
cx q[22],q[23];
cx q[9],q[24];
cx q[10],q[24];
cx q[11],q[24];
cx q[12],q[24];
cx q[13],q[24];
cx q[14],q[24];
cx q[15],q[24];
cx q[16],q[24];
cx q[17],q[24];
cx q[18],q[24];
cx q[19],q[24];
cx q[20],q[24];
cx q[21],q[24];
cx q[22],q[24];
cx q[23],q[24];
cx q[9],q[25];
cx q[10],q[25];
cx q[11],q[25];
cx q[12],q[25];
cx q[13],q[25];
cx q[14],q[25];
cx q[15],q[25];
cx q[16],q[25];
cx q[17],q[25];
cx q[18],q[25];
cx q[19],q[25];
cx q[20],q[25];
cx q[21],q[25];
cx q[22],q[25];
cx q[23],q[25];
cx q[24],q[25];
cx q[9],q[26];
cx q[10],q[26];
cx q[11],q[26];
cx q[12],q[26];
cx q[13],q[26];
cx q[14],q[26];
cx q[15],q[26];
cx q[16],q[26];
cx q[17],q[26];
cx q[18],q[26];
cx q[19],q[26];
cx q[20],q[26];
cx q[21],q[26];
cx q[22],q[26];
cx q[23],q[26];
cx q[24],q[26];
cx q[25],q[26];
cx q[9],q[27];
cx q[10],q[27];
cx q[11],q[27];
cx q[12],q[27];
cx q[13],q[27];
cx q[14],q[27];
cx q[15],q[27];
cx q[16],q[27];
cx q[17],q[27];
cx q[18],q[27];
cx q[19],q[27];
cx q[20],q[27];
cx q[21],q[27];
cx q[22],q[27];
cx q[23],q[27];
cx q[24],q[27];
cx q[25],q[27];
cx q[26],q[27];
cx q[9],q[28];
cx q[10],q[28];
cx q[11],q[28];
cx q[12],q[28];
cx q[13],q[28];
cx q[14],q[28];
cx q[15],q[28];
cx q[16],q[28];
cx q[17],q[28];
cx q[18],q[28];
cx q[19],q[28];
cx q[20],q[28];
cx q[21],q[28];
cx q[22],q[28];
cx q[23],q[28];
cx q[24],q[28];
cx q[25],q[28];
cx q[26],q[28];
cx q[27],q[28];
cx q[9],q[29];
cx q[10],q[29];
cx q[11],q[29];
cx q[12],q[29];
cx q[13],q[29];
cx q[14],q[29];
cx q[15],q[29];
cx q[16],q[29];
cx q[17],q[29];
cx q[18],q[29];
cx q[19],q[29];
cx q[20],q[29];
cx q[21],q[29];
cx q[22],q[29];
cx q[23],q[29];
cx q[24],q[29];
cx q[25],q[29];
cx q[26],q[29];
cx q[27],q[29];
cx q[28],q[29];
cx q[9],q[30];
cx q[10],q[30];
cx q[11],q[30];
cx q[12],q[30];
cx q[13],q[30];
cx q[14],q[30];
cx q[15],q[30];
cx q[16],q[30];
cx q[17],q[30];
cx q[18],q[30];
cx q[19],q[30];
cx q[20],q[30];
cx q[21],q[30];
cx q[22],q[30];
cx q[23],q[30];
cx q[24],q[30];
cx q[25],q[30];
cx q[26],q[30];
cx q[27],q[30];
cx q[28],q[30];
cx q[29],q[30];
cx q[9],q[31];
cx q[10],q[31];
cx q[11],q[31];
cx q[12],q[31];
cx q[13],q[31];
cx q[14],q[31];
cx q[15],q[31];
cx q[16],q[31];
cx q[17],q[31];
cx q[18],q[31];
cx q[19],q[31];
cx q[20],q[31];
cx q[21],q[31];
cx q[22],q[31];
cx q[23],q[31];
cx q[24],q[31];
cx q[25],q[31];
cx q[26],q[31];
cx q[27],q[31];
cx q[28],q[31];
cx q[29],q[31];
cx q[30],q[31];
cx q[9],q[32];
cx q[10],q[32];
cx q[11],q[32];
cx q[12],q[32];
cx q[13],q[32];
cx q[14],q[32];
cx q[15],q[32];
cx q[16],q[32];
cx q[17],q[32];
cx q[18],q[32];
cx q[19],q[32];
cx q[20],q[32];
cx q[21],q[32];
cx q[22],q[32];
cx q[23],q[32];
cx q[24],q[32];
cx q[25],q[32];
cx q[26],q[32];
cx q[27],q[32];
cx q[28],q[32];
cx q[29],q[32];
cx q[30],q[32];
cx q[31],q[32];
cx q[9],q[33];
cx q[10],q[33];
cx q[11],q[33];
cx q[12],q[33];
cx q[13],q[33];
cx q[14],q[33];
cx q[15],q[33];
cx q[16],q[33];
cx q[17],q[33];
cx q[18],q[33];
cx q[19],q[33];
cx q[20],q[33];
cx q[21],q[33];
cx q[22],q[33];
cx q[23],q[33];
cx q[24],q[33];
cx q[25],q[33];
cx q[26],q[33];
cx q[27],q[33];
cx q[28],q[33];
cx q[29],q[33];
cx q[30],q[33];
cx q[31],q[33];
cx q[32],q[33];
cx q[9],q[34];
cx q[10],q[34];
cx q[11],q[34];
cx q[12],q[34];
cx q[13],q[34];
cx q[14],q[34];
cx q[15],q[34];
cx q[16],q[34];
cx q[17],q[34];
cx q[18],q[34];
cx q[19],q[34];
cx q[20],q[34];
cx q[21],q[34];
cx q[22],q[34];
cx q[23],q[34];
cx q[24],q[34];
cx q[25],q[34];
cx q[26],q[34];
cx q[27],q[34];
cx q[28],q[34];
cx q[29],q[34];
cx q[30],q[34];
cx q[31],q[34];
cx q[32],q[34];
cx q[33],q[34];
cx q[9],q[35];
cx q[10],q[35];
cx q[11],q[35];
cx q[12],q[35];
cx q[13],q[35];
cx q[14],q[35];
cx q[15],q[35];
cx q[16],q[35];
cx q[17],q[35];
cx q[18],q[35];
cx q[19],q[35];
cx q[20],q[35];
cx q[21],q[35];
cx q[22],q[35];
cx q[23],q[35];
cx q[24],q[35];
cx q[25],q[35];
cx q[26],q[35];
cx q[27],q[35];
cx q[28],q[35];
cx q[29],q[35];
cx q[30],q[35];
cx q[31],q[35];
cx q[32],q[35];
cx q[33],q[35];
cx q[34],q[35];
cx q[9],q[36];
cx q[10],q[36];
cx q[11],q[36];
cx q[12],q[36];
cx q[13],q[36];
cx q[14],q[36];
cx q[15],q[36];
cx q[16],q[36];
cx q[17],q[36];
cx q[18],q[36];
cx q[19],q[36];
cx q[20],q[36];
cx q[21],q[36];
cx q[22],q[36];
cx q[23],q[36];
cx q[24],q[36];
cx q[25],q[36];
cx q[26],q[36];
cx q[27],q[36];
cx q[28],q[36];
cx q[29],q[36];
cx q[30],q[36];
cx q[31],q[36];
cx q[32],q[36];
cx q[33],q[36];
cx q[34],q[36];
cx q[35],q[36];
cx q[9],q[37];
cx q[10],q[37];
ry(1.9041982282388) q[10];
cx q[11],q[37];
ry(1.52100758355808) q[11];
cx q[12],q[37];
ry(3.50336708260893) q[12];
cx q[13],q[37];
ry(3.55318539846825) q[13];
cx q[14],q[37];
ry(2.98534395592546) q[14];
cx q[15],q[37];
ry(1.83970394259416) q[15];
cx q[16],q[37];
ry(0.403701320528425) q[16];
cx q[17],q[37];
ry(6.15010207501049) q[17];
cx q[18],q[37];
ry(2.13444733187909) q[18];
cx q[19],q[37];
ry(3.11048228390001) q[19];
cx q[20],q[37];
ry(6.13917926104576) q[20];
cx q[21],q[37];
ry(2.76946362040522) q[21];
cx q[22],q[37];
ry(1.99976701506017) q[22];
cx q[23],q[37];
ry(3.26598078436842) q[23];
cx q[24],q[37];
ry(3.63253832178278) q[24];
cx q[25],q[37];
ry(5.36542399444941) q[25];
cx q[26],q[37];
ry(0.427867788552637) q[26];
cx q[27],q[37];
ry(2.91873314617137) q[27];
cx q[28],q[37];
ry(4.91313121306986) q[28];
cx q[29],q[37];
ry(4.51511461989171) q[29];
cx q[30],q[37];
ry(3.6820846947544) q[30];
cx q[31],q[37];
ry(0.23307107221288) q[31];
cx q[32],q[37];
ry(2.2032390855788) q[32];
cx q[33],q[37];
ry(3.53863143394523) q[33];
cx q[34],q[37];
ry(1.88325833054083) q[34];
cx q[35],q[37];
ry(3.21909042421466) q[35];
cx q[36],q[37];
ry(4.23151748982037) q[36];
ry(1.0002437265563) q[37];
ry(0.15959153738279) q[9];
cx q[0],q[9];
cx q[0],q[10];
cx q[0],q[11];
cx q[0],q[12];
cx q[0],q[13];
cx q[0],q[14];
cx q[0],q[15];
cx q[0],q[16];
cx q[0],q[17];
cx q[0],q[18];
cx q[0],q[19];
cx q[0],q[20];
cx q[0],q[21];
cx q[0],q[22];
cx q[0],q[23];
cx q[0],q[24];
cx q[0],q[25];
cx q[0],q[26];
cx q[0],q[27];
cx q[0],q[28];
cx q[0],q[29];
cx q[0],q[30];
cx q[0],q[31];
cx q[0],q[32];
cx q[0],q[33];
cx q[0],q[34];
cx q[0],q[35];
cx q[0],q[36];
cx q[0],q[37];
ry(0.317160555452255) q[0];
cx q[1],q[9];
cx q[1],q[10];
cx q[1],q[11];
cx q[1],q[12];
cx q[1],q[13];
cx q[1],q[14];
cx q[1],q[15];
cx q[1],q[16];
cx q[1],q[17];
cx q[1],q[18];
cx q[1],q[19];
cx q[1],q[20];
cx q[1],q[21];
cx q[1],q[22];
cx q[1],q[23];
cx q[1],q[24];
cx q[1],q[25];
cx q[1],q[26];
cx q[1],q[27];
cx q[1],q[28];
cx q[1],q[29];
cx q[1],q[30];
cx q[1],q[31];
cx q[1],q[32];
cx q[1],q[33];
cx q[1],q[34];
cx q[1],q[35];
cx q[1],q[36];
cx q[1],q[37];
ry(2.12255981813663) q[1];
cx q[2],q[9];
cx q[2],q[10];
cx q[2],q[11];
cx q[2],q[12];
cx q[2],q[13];
cx q[2],q[14];
cx q[2],q[15];
cx q[2],q[16];
cx q[2],q[17];
cx q[2],q[18];
cx q[2],q[19];
cx q[2],q[20];
cx q[2],q[21];
cx q[2],q[22];
cx q[2],q[23];
cx q[2],q[24];
cx q[2],q[25];
cx q[2],q[26];
cx q[2],q[27];
cx q[2],q[28];
cx q[2],q[29];
cx q[2],q[30];
cx q[2],q[31];
cx q[2],q[32];
cx q[2],q[33];
cx q[2],q[34];
cx q[2],q[35];
cx q[2],q[36];
cx q[2],q[37];
ry(0.67898470936625) q[2];
cx q[3],q[9];
cx q[3],q[10];
cx q[3],q[11];
cx q[3],q[12];
cx q[3],q[13];
cx q[3],q[14];
cx q[3],q[15];
cx q[3],q[16];
cx q[3],q[17];
cx q[3],q[18];
cx q[3],q[19];
cx q[3],q[20];
cx q[3],q[21];
cx q[3],q[22];
cx q[3],q[23];
cx q[3],q[24];
cx q[3],q[25];
cx q[3],q[26];
cx q[3],q[27];
cx q[3],q[28];
cx q[3],q[29];
cx q[3],q[30];
cx q[3],q[31];
cx q[3],q[32];
cx q[3],q[33];
cx q[3],q[34];
cx q[3],q[35];
cx q[3],q[36];
cx q[3],q[37];
ry(1.12407949822704) q[3];
cx q[4],q[9];
cx q[4],q[10];
cx q[4],q[11];
cx q[4],q[12];
cx q[4],q[13];
cx q[4],q[14];
cx q[4],q[15];
cx q[4],q[16];
cx q[4],q[17];
cx q[4],q[18];
cx q[4],q[19];
cx q[4],q[20];
cx q[4],q[21];
cx q[4],q[22];
cx q[4],q[23];
cx q[4],q[24];
cx q[4],q[25];
cx q[4],q[26];
cx q[4],q[27];
cx q[4],q[28];
cx q[4],q[29];
cx q[4],q[30];
cx q[4],q[31];
cx q[4],q[32];
cx q[4],q[33];
cx q[4],q[34];
cx q[4],q[35];
cx q[4],q[36];
cx q[4],q[37];
ry(5.56581579534249) q[4];
cx q[5],q[9];
cx q[5],q[10];
cx q[5],q[11];
cx q[5],q[12];
cx q[5],q[13];
cx q[5],q[14];
cx q[5],q[15];
cx q[5],q[16];
cx q[5],q[17];
cx q[5],q[18];
cx q[5],q[19];
cx q[5],q[20];
cx q[5],q[21];
cx q[5],q[22];
cx q[5],q[23];
cx q[5],q[24];
cx q[5],q[25];
cx q[5],q[26];
cx q[5],q[27];
cx q[5],q[28];
cx q[5],q[29];
cx q[5],q[30];
cx q[5],q[31];
cx q[5],q[32];
cx q[5],q[33];
cx q[5],q[34];
cx q[5],q[35];
cx q[5],q[36];
cx q[5],q[37];
ry(2.29565581889063) q[5];
cx q[6],q[9];
cx q[6],q[10];
cx q[6],q[11];
cx q[6],q[12];
cx q[6],q[13];
cx q[6],q[14];
cx q[6],q[15];
cx q[6],q[16];
cx q[6],q[17];
cx q[6],q[18];
cx q[6],q[19];
cx q[6],q[20];
cx q[6],q[21];
cx q[6],q[22];
cx q[6],q[23];
cx q[6],q[24];
cx q[6],q[25];
cx q[6],q[26];
cx q[6],q[27];
cx q[6],q[28];
cx q[6],q[29];
cx q[6],q[30];
cx q[6],q[31];
cx q[6],q[32];
cx q[6],q[33];
cx q[6],q[34];
cx q[6],q[35];
cx q[6],q[36];
cx q[6],q[37];
ry(1.37456836042611) q[6];
cx q[7],q[9];
cx q[7],q[10];
cx q[7],q[11];
cx q[7],q[12];
cx q[7],q[13];
cx q[7],q[14];
cx q[7],q[15];
cx q[7],q[16];
cx q[7],q[17];
cx q[7],q[18];
cx q[7],q[19];
cx q[7],q[20];
cx q[7],q[21];
cx q[7],q[22];
cx q[7],q[23];
cx q[7],q[24];
cx q[7],q[25];
cx q[7],q[26];
cx q[7],q[27];
cx q[7],q[28];
cx q[7],q[29];
cx q[7],q[30];
cx q[7],q[31];
cx q[7],q[32];
cx q[7],q[33];
cx q[7],q[34];
cx q[7],q[35];
cx q[7],q[36];
cx q[7],q[37];
ry(4.72807288042643) q[7];
cx q[8],q[9];
cx q[8],q[10];
cx q[8],q[11];
cx q[8],q[12];
cx q[8],q[13];
cx q[8],q[14];
cx q[8],q[15];
cx q[8],q[16];
cx q[8],q[17];
cx q[8],q[18];
cx q[8],q[19];
cx q[8],q[20];
cx q[8],q[21];
cx q[8],q[22];
cx q[8],q[23];
cx q[8],q[24];
cx q[8],q[25];
cx q[8],q[26];
cx q[8],q[27];
cx q[8],q[28];
cx q[8],q[29];
cx q[8],q[30];
cx q[8],q[31];
cx q[8],q[32];
cx q[8],q[33];
cx q[8],q[34];
cx q[8],q[35];
cx q[8],q[36];
cx q[8],q[37];
ry(0.671544234299134) q[8];
cx q[9],q[10];
cx q[9],q[11];
cx q[10],q[11];
cx q[9],q[12];
cx q[10],q[12];
cx q[11],q[12];
cx q[9],q[13];
cx q[10],q[13];
cx q[11],q[13];
cx q[12],q[13];
cx q[9],q[14];
cx q[10],q[14];
cx q[11],q[14];
cx q[12],q[14];
cx q[13],q[14];
cx q[9],q[15];
cx q[10],q[15];
cx q[11],q[15];
cx q[12],q[15];
cx q[13],q[15];
cx q[14],q[15];
cx q[9],q[16];
cx q[10],q[16];
cx q[11],q[16];
cx q[12],q[16];
cx q[13],q[16];
cx q[14],q[16];
cx q[15],q[16];
cx q[9],q[17];
cx q[10],q[17];
cx q[11],q[17];
cx q[12],q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[15],q[17];
cx q[16],q[17];
cx q[9],q[18];
cx q[10],q[18];
cx q[11],q[18];
cx q[12],q[18];
cx q[13],q[18];
cx q[14],q[18];
cx q[15],q[18];
cx q[16],q[18];
cx q[17],q[18];
cx q[9],q[19];
cx q[10],q[19];
cx q[11],q[19];
cx q[12],q[19];
cx q[13],q[19];
cx q[14],q[19];
cx q[15],q[19];
cx q[16],q[19];
cx q[17],q[19];
cx q[18],q[19];
cx q[9],q[20];
cx q[10],q[20];
cx q[11],q[20];
cx q[12],q[20];
cx q[13],q[20];
cx q[14],q[20];
cx q[15],q[20];
cx q[16],q[20];
cx q[17],q[20];
cx q[18],q[20];
cx q[19],q[20];
cx q[9],q[21];
cx q[10],q[21];
cx q[11],q[21];
cx q[12],q[21];
cx q[13],q[21];
cx q[14],q[21];
cx q[15],q[21];
cx q[16],q[21];
cx q[17],q[21];
cx q[18],q[21];
cx q[19],q[21];
cx q[20],q[21];
cx q[9],q[22];
cx q[10],q[22];
cx q[11],q[22];
cx q[12],q[22];
cx q[13],q[22];
cx q[14],q[22];
cx q[15],q[22];
cx q[16],q[22];
cx q[17],q[22];
cx q[18],q[22];
cx q[19],q[22];
cx q[20],q[22];
cx q[21],q[22];
cx q[9],q[23];
cx q[10],q[23];
cx q[11],q[23];
cx q[12],q[23];
cx q[13],q[23];
cx q[14],q[23];
cx q[15],q[23];
cx q[16],q[23];
cx q[17],q[23];
cx q[18],q[23];
cx q[19],q[23];
cx q[20],q[23];
cx q[21],q[23];
cx q[22],q[23];
cx q[9],q[24];
cx q[10],q[24];
cx q[11],q[24];
cx q[12],q[24];
cx q[13],q[24];
cx q[14],q[24];
cx q[15],q[24];
cx q[16],q[24];
cx q[17],q[24];
cx q[18],q[24];
cx q[19],q[24];
cx q[20],q[24];
cx q[21],q[24];
cx q[22],q[24];
cx q[23],q[24];
cx q[9],q[25];
cx q[10],q[25];
cx q[11],q[25];
cx q[12],q[25];
cx q[13],q[25];
cx q[14],q[25];
cx q[15],q[25];
cx q[16],q[25];
cx q[17],q[25];
cx q[18],q[25];
cx q[19],q[25];
cx q[20],q[25];
cx q[21],q[25];
cx q[22],q[25];
cx q[23],q[25];
cx q[24],q[25];
cx q[9],q[26];
cx q[10],q[26];
cx q[11],q[26];
cx q[12],q[26];
cx q[13],q[26];
cx q[14],q[26];
cx q[15],q[26];
cx q[16],q[26];
cx q[17],q[26];
cx q[18],q[26];
cx q[19],q[26];
cx q[20],q[26];
cx q[21],q[26];
cx q[22],q[26];
cx q[23],q[26];
cx q[24],q[26];
cx q[25],q[26];
cx q[9],q[27];
cx q[10],q[27];
cx q[11],q[27];
cx q[12],q[27];
cx q[13],q[27];
cx q[14],q[27];
cx q[15],q[27];
cx q[16],q[27];
cx q[17],q[27];
cx q[18],q[27];
cx q[19],q[27];
cx q[20],q[27];
cx q[21],q[27];
cx q[22],q[27];
cx q[23],q[27];
cx q[24],q[27];
cx q[25],q[27];
cx q[26],q[27];
cx q[9],q[28];
cx q[10],q[28];
cx q[11],q[28];
cx q[12],q[28];
cx q[13],q[28];
cx q[14],q[28];
cx q[15],q[28];
cx q[16],q[28];
cx q[17],q[28];
cx q[18],q[28];
cx q[19],q[28];
cx q[20],q[28];
cx q[21],q[28];
cx q[22],q[28];
cx q[23],q[28];
cx q[24],q[28];
cx q[25],q[28];
cx q[26],q[28];
cx q[27],q[28];
cx q[9],q[29];
cx q[10],q[29];
cx q[11],q[29];
cx q[12],q[29];
cx q[13],q[29];
cx q[14],q[29];
cx q[15],q[29];
cx q[16],q[29];
cx q[17],q[29];
cx q[18],q[29];
cx q[19],q[29];
cx q[20],q[29];
cx q[21],q[29];
cx q[22],q[29];
cx q[23],q[29];
cx q[24],q[29];
cx q[25],q[29];
cx q[26],q[29];
cx q[27],q[29];
cx q[28],q[29];
cx q[9],q[30];
cx q[10],q[30];
cx q[11],q[30];
cx q[12],q[30];
cx q[13],q[30];
cx q[14],q[30];
cx q[15],q[30];
cx q[16],q[30];
cx q[17],q[30];
cx q[18],q[30];
cx q[19],q[30];
cx q[20],q[30];
cx q[21],q[30];
cx q[22],q[30];
cx q[23],q[30];
cx q[24],q[30];
cx q[25],q[30];
cx q[26],q[30];
cx q[27],q[30];
cx q[28],q[30];
cx q[29],q[30];
cx q[9],q[31];
cx q[10],q[31];
cx q[11],q[31];
cx q[12],q[31];
cx q[13],q[31];
cx q[14],q[31];
cx q[15],q[31];
cx q[16],q[31];
cx q[17],q[31];
cx q[18],q[31];
cx q[19],q[31];
cx q[20],q[31];
cx q[21],q[31];
cx q[22],q[31];
cx q[23],q[31];
cx q[24],q[31];
cx q[25],q[31];
cx q[26],q[31];
cx q[27],q[31];
cx q[28],q[31];
cx q[29],q[31];
cx q[30],q[31];
cx q[9],q[32];
cx q[10],q[32];
cx q[11],q[32];
cx q[12],q[32];
cx q[13],q[32];
cx q[14],q[32];
cx q[15],q[32];
cx q[16],q[32];
cx q[17],q[32];
cx q[18],q[32];
cx q[19],q[32];
cx q[20],q[32];
cx q[21],q[32];
cx q[22],q[32];
cx q[23],q[32];
cx q[24],q[32];
cx q[25],q[32];
cx q[26],q[32];
cx q[27],q[32];
cx q[28],q[32];
cx q[29],q[32];
cx q[30],q[32];
cx q[31],q[32];
cx q[9],q[33];
cx q[10],q[33];
cx q[11],q[33];
cx q[12],q[33];
cx q[13],q[33];
cx q[14],q[33];
cx q[15],q[33];
cx q[16],q[33];
cx q[17],q[33];
cx q[18],q[33];
cx q[19],q[33];
cx q[20],q[33];
cx q[21],q[33];
cx q[22],q[33];
cx q[23],q[33];
cx q[24],q[33];
cx q[25],q[33];
cx q[26],q[33];
cx q[27],q[33];
cx q[28],q[33];
cx q[29],q[33];
cx q[30],q[33];
cx q[31],q[33];
cx q[32],q[33];
cx q[9],q[34];
cx q[10],q[34];
cx q[11],q[34];
cx q[12],q[34];
cx q[13],q[34];
cx q[14],q[34];
cx q[15],q[34];
cx q[16],q[34];
cx q[17],q[34];
cx q[18],q[34];
cx q[19],q[34];
cx q[20],q[34];
cx q[21],q[34];
cx q[22],q[34];
cx q[23],q[34];
cx q[24],q[34];
cx q[25],q[34];
cx q[26],q[34];
cx q[27],q[34];
cx q[28],q[34];
cx q[29],q[34];
cx q[30],q[34];
cx q[31],q[34];
cx q[32],q[34];
cx q[33],q[34];
cx q[9],q[35];
cx q[10],q[35];
cx q[11],q[35];
cx q[12],q[35];
cx q[13],q[35];
cx q[14],q[35];
cx q[15],q[35];
cx q[16],q[35];
cx q[17],q[35];
cx q[18],q[35];
cx q[19],q[35];
cx q[20],q[35];
cx q[21],q[35];
cx q[22],q[35];
cx q[23],q[35];
cx q[24],q[35];
cx q[25],q[35];
cx q[26],q[35];
cx q[27],q[35];
cx q[28],q[35];
cx q[29],q[35];
cx q[30],q[35];
cx q[31],q[35];
cx q[32],q[35];
cx q[33],q[35];
cx q[34],q[35];
cx q[9],q[36];
cx q[10],q[36];
cx q[11],q[36];
cx q[12],q[36];
cx q[13],q[36];
cx q[14],q[36];
cx q[15],q[36];
cx q[16],q[36];
cx q[17],q[36];
cx q[18],q[36];
cx q[19],q[36];
cx q[20],q[36];
cx q[21],q[36];
cx q[22],q[36];
cx q[23],q[36];
cx q[24],q[36];
cx q[25],q[36];
cx q[26],q[36];
cx q[27],q[36];
cx q[28],q[36];
cx q[29],q[36];
cx q[30],q[36];
cx q[31],q[36];
cx q[32],q[36];
cx q[33],q[36];
cx q[34],q[36];
cx q[35],q[36];
cx q[9],q[37];
cx q[10],q[37];
ry(2.95174805327437) q[10];
cx q[11],q[37];
ry(3.75895124371776) q[11];
cx q[12],q[37];
ry(0.927525023210008) q[12];
cx q[13],q[37];
ry(1.1563248901851) q[13];
cx q[14],q[37];
ry(4.05310770709635) q[14];
cx q[15],q[37];
ry(0.305538774471667) q[15];
cx q[16],q[37];
ry(1.56207845620743) q[16];
cx q[17],q[37];
ry(3.40805321965288) q[17];
cx q[18],q[37];
ry(1.42485893849411) q[18];
cx q[19],q[37];
ry(2.3964793521016) q[19];
cx q[20],q[37];
ry(5.79455949647164) q[20];
cx q[21],q[37];
ry(5.81418870750046) q[21];
cx q[22],q[37];
ry(3.56099479893478) q[22];
cx q[23],q[37];
ry(3.35189642637099) q[23];
cx q[24],q[37];
ry(0.0933682884398256) q[24];
cx q[25],q[37];
ry(6.14432228370922) q[25];
cx q[26],q[37];
ry(3.60044679041053) q[26];
cx q[27],q[37];
ry(4.97475592586193) q[27];
cx q[28],q[37];
ry(3.52836895522703) q[28];
cx q[29],q[37];
ry(5.51245989927165) q[29];
cx q[30],q[37];
ry(3.67061064633955) q[30];
cx q[31],q[37];
ry(4.45383481403822) q[31];
cx q[32],q[37];
ry(0.933263199187574) q[32];
cx q[33],q[37];
ry(2.69203538793332) q[33];
cx q[34],q[37];
ry(4.35983986964051) q[34];
cx q[35],q[37];
ry(0.657345241626873) q[35];
cx q[36],q[37];
ry(2.7621211709445) q[36];
ry(1.04427889247882) q[37];
ry(4.67848014211931) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26],q[27],q[28],q[29],q[30],q[31],q[32],q[33],q[34],q[35],q[36],q[37];
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
meas[22] = measure q[22];
meas[23] = measure q[23];
meas[24] = measure q[24];
meas[25] = measure q[25];
meas[26] = measure q[26];
meas[27] = measure q[27];
meas[28] = measure q[28];
meas[29] = measure q[29];
meas[30] = measure q[30];
meas[31] = measure q[31];
meas[32] = measure q[32];
meas[33] = measure q[33];
meas[34] = measure q[34];
meas[35] = measure q[35];
meas[36] = measure q[36];
meas[37] = measure q[37];