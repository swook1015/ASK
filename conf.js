/**
 * Created by Il Yeup, Ahn in KETI on 2017-02-23.
 */

/**
 * Copyright (c) 2018, OCEAN
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products derived from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

const ip = require("ip");
const { nanoid } = require("nanoid");

let conf = {};
let cse = {};
let ae = {};
let cnt_arr = [];
let sub_arr = [];
let acp = {};

conf.useprotocol = 'http'; // select one for 'http' or 'mqtt' or 'coap' or 'ws'

conf.sim = 'disable'; // enable or disable

// build cse
cse = {
    host    : '192.168.0.2',
    port    : '7579',
    name    : 'Mobius',
    id      : '/Mobius2',
    mqttport: '1883',
    wsport  : '7577',
};

conf.cameras = {};

// build ae
let ae_name = 'KETI3_DEMO';

ae = {
    name    : ae_name,
    id      : 'S' + ae_name,
    parent  : '/' + cse.name,
    appid   : 'fall_detection',
    port    : '9727',
    bodytype: 'json',
    tasport : '3105',
};

// build cnt
var count = 0;
cnt_arr = [
    {
        parent: '/' + cse.name + '/' + ae.name,
        name: 'fall_state',   // 정상 상태
    },
    {
        parent: '/' + cse.name + '/' + ae.name,
        name: 'fall_event',   // 낙상 발생
    },
    {
        parent: '/' + cse.name + '/' + ae.name,
        name: 'fall_clip',   // 낙상 영상 클립 메타(경로 등)
    },
];

exports.clip = {
    host: "192.168.0.2",
    port: 8081
};
// build sub
sub_arr = [];

// for tas
let tas = {
    client: {
        connected: false,
    },

    connection: {
        host: 'localhost',
        port: 1883,
        endpoint: '',
        clean: true,
        connectTimeout: 4000,
        reconnectPeriod: 4000,
        clientId: 'thyme_' + nanoid(15),
        username: 'keti_thyme',
        password: 'keti_thyme',
    },
};

// build acp: not complete
acp.parent = '/' + cse.name + '/' + ae.name;
acp.name = 'acp-' + ae.name;
acp.id = ae.id;

conf.usesecure  = 'disable';

if(conf.usesecure === 'enable') {
    cse.mqttport = '8883';
}

conf.cse = cse;
conf.ae = ae;
conf.cnt = cnt_arr;
conf.sub = sub_arr;
conf.acp = acp;
conf.tas = tas;

module.exports = conf;
