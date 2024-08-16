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

qubit[6] q;
bit[6] meas;
bit[6] meas3;
u2(0.0034559680191952324,-pi) q[0];
u2(-0.0005051053317757237,-pi) q[1];
rzz(0.30640835800234384) q[0],q[1];
u2(-0.00029925130246333254,-pi) q[2];
rzz(0.3064578071108339) q[0],q[2];
rzz(0.30641123969344364) q[1],q[2];
u2(-0.0006473901644978852,-pi) q[3];
rzz(0.3064382202982792) q[0],q[3];
rzz(0.3064321236130968) q[1],q[3];
rzz(0.30643112647579906) q[2],q[3];
u2(-0.012870739783510388,-pi) q[4];
rzz(0.3061241270907537) q[0],q[4];
rzz(0.3064529865522876) q[1],q[4];
rzz(0.3062438001194054) q[2],q[4];
rzz(0.3064907532354401) q[3],q[4];
u2(-0.0010250221512837498,-pi) q[5];
rzz(0.3064297198462831) q[0],q[5];
u3(0.49809428163893216,1.6109783174340429,-pi/2) q[0];
rzz(0.30645277488492745) q[1],q[5];
u3(0.49809428163893216,1.5649235474171732,-pi/2) q[1];
rzz(3.5625612576914834) q[0],q[1];
rzz(0.3063943933113396) q[2],q[5];
u3(0.49809428163893216,1.5673169794839428,-pi/2) q[2];
rzz(3.5631361946131874) q[0],q[2];
rzz(3.5625947626556944) q[1],q[2];
rzz(0.30645035976402574) q[3],q[5];
u3(0.49809428163893216,1.5632692242912736,-pi/2) q[3];
rzz(3.5629084618580364) q[0],q[3];
rzz(3.5628375766688305) q[1],q[3];
rzz(3.5628259831120213) q[2],q[3];
rzz(0.3063413283269469) q[4],q[5];
u3(0.4980942816389322,1.4211502818583153,-pi/2) q[4];
rzz(3.559256550076877) q[0],q[4];
rzz(3.563080146745467) q[1],q[4];
rzz(3.5606479693522672) q[2],q[4];
rzz(3.563519253965348) q[3],q[4];
u3(0.49809428163893216,1.558878557226711,-pi/2) q[5];
rzz(3.5628096284543016) q[0],q[5];
u3(3.02922917772048,1.5800774044167776,-pi/2) q[0];
rzz(3.5630776857227273) q[1],q[5];
u3(3.02922917772048,1.5694398553908213,-pi/2) q[1];
rzz(0.8228663398554805) q[0],q[1];
rzz(3.5623988924496484) q[2],q[5];
u3(3.02922917772048,1.5699926808840594,-pi/2) q[2];
rzz(0.8229991365166998) q[0],q[2];
rzz(0.8228740786998885) q[1],q[2];
rzz(3.563049605496022) q[3],q[5];
u3(3.02922917772048,1.5690577463638933,-pi/2) q[3];
rzz(0.8229465357036495) q[0],q[3];
rzz(0.822930162922391) q[1],q[3];
rzz(0.8229274850883925) q[2],q[3];
rzz(3.5617819143137712) q[4],q[5];
u3(3.02922917772048,1.53623167390508,-pi/2) q[4];
rzz(0.8221030315325001) q[0],q[4];
rzz(0.8229861908014034) q[1],q[4];
rzz(0.8224244160655724) q[2],q[4];
rzz(0.8230876140541399) q[3],q[4];
u3(3.02922917772048,1.5680436074128394,-pi/2) q[5];
rzz(0.8229237075540952) q[0],q[5];
rx(6.436654741410069) q[0];
rzz(0.8229856223640832) q[1],q[5];
rx(6.436654741410069) q[1];
rzz(0.8228288373726867) q[2],q[5];
rx(6.436654741410069) q[2];
rzz(0.8229791365041359) q[3],q[5];
rx(6.436654741410069) q[3];
rzz(0.8226863301977311) q[4],q[5];
rx(6.436654741410069) q[4];
rx(6.436654741410069) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];
meas[5] = measure q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
meas3[0] = measure q[0];
meas3[1] = measure q[1];
meas3[2] = measure q[2];
meas3[3] = measure q[3];
meas3[4] = measure q[4];
meas3[5] = measure q[5];