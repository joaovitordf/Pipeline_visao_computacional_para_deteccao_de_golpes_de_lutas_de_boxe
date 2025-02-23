import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WebRTC App',
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  RTCVideoRenderer _localRenderer = RTCVideoRenderer();
  MediaStream? _localStream;
  RTCPeerConnection? _peerConnection;

  @override
  void initState() {
    super.initState();
    _initializeRenderers();
    _initLocalStream();
    _createPeerConnection();
  }

  @override
  void dispose() {
    _localRenderer.dispose();
    _localStream?.dispose();
    _peerConnection?.close();
    super.dispose();
  }

  // Inicializa o renderizador de vídeo
  Future<void> _initializeRenderers() async {
    await _localRenderer.initialize();
  }

  // Captura o stream da câmera traseira, sem áudio
  Future<void> _initLocalStream() async {
    final Map<String, dynamic> mediaConstraints = {
      'audio': false,
      'video': {
        'facingMode': 'environment', // utiliza a câmera traseira
      },
    };

    try {
      MediaStream stream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
      setState(() {
        _localStream = stream;
        _localRenderer.srcObject = _localStream;
      });
    } catch (e) {
      print("Erro ao acessar a câmera: $e");
    }
  }

  // Cria a conexão WebRTC, adiciona o stream local e gera a oferta SDP
  Future<void> _createPeerConnection() async {
    Map<String, dynamic> configuration = {
      'iceServers': [
        {'urls': 'stun:stun.l.google.com:19302'},
      ]
    };

    // Restrições para a oferta SDP
    final Map<String, dynamic> offerSdpConstraints = {
      'mandatory': {
        'OfferToReceiveAudio': false,
        'OfferToReceiveVideo': true,
      },
      'optional': [],
    };

    try {
      _peerConnection = await createPeerConnection(configuration, offerSdpConstraints);

      // Adiciona cada faixa do stream local à conexão
      if (_localStream != null) {
        _localStream!.getTracks().forEach((track) {
          _peerConnection!.addTrack(track, _localStream!);
        });
      }

      // Cria a oferta SDP
      RTCSessionDescription description = await _peerConnection!.createOffer(offerSdpConstraints);
      await _peerConnection!.setLocalDescription(description);

      // Serializa a oferta SDP (exemplo em JSON)
      final offerData = {
        'sdp': description.sdp,
        'type': description.type,
      };
      print("Oferta SDP serializada: $offerData");

      // TODO: Implemente a comunicação com o servidor (via WebSocket)

    } catch (e) {
      print("Erro ao criar a conexão WebRTC: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('WebRTC App'),
      ),
      body: Center(
        // Exibe o vídeo capturado localmente
        child: RTCVideoView(_localRenderer),
      ),
    );
  }
}
