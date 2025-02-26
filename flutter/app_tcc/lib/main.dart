import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:flutter/scheduler.dart';

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
  Socket? _socket;
  GlobalKey _videoKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    _initializeRenderers();
    _initLocalStream();
    _connectToServer();
  }

  @override
  void dispose() {
    _localRenderer.dispose();
    _localStream?.dispose();
    _socket?.close();
    super.dispose();
  }

  Future<void> _initializeRenderers() async {
    await _localRenderer.initialize();
  }

  Future<void> _initLocalStream() async {
    final Map<String, dynamic> mediaConstraints = {
      'audio': false,
      'video': {
        'mandatory': {
          'minWidth': '1280',
          'minHeight': '720',
          'minFrameRate': '15',
        },
        'facingMode': 'environment',
      },
    };

    try {
      MediaStream stream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
      setState(() {
        _localStream = stream;
        _localRenderer.srcObject = _localStream;
      });
      _startFrameCapture();
    } catch (e) {
      print("Erro ao acessar a câmera: $e");
    }
  }

  WebSocket? _webSocket;

  Future<void> _connectToServer() async {
    try {
      _webSocket = await WebSocket.connect('ws://192.168.1.104:8765');
      print('Conectado ao servidor via WebSocket.');

      // Exemplo de como ouvir mensagens do servidor:
      _webSocket!.listen((data) {
        print('Mensagem recebida: $data');
      });

    } catch (e) {
      print('Erro ao conectar ao servidor via WebSocket: $e');
    }
  }

  void _startFrameCapture() {
    SchedulerBinding.instance.addPostFrameCallback((_) async {
      if (_webSocket != null && _videoKey.currentContext != null) {
        RenderRepaintBoundary boundary =
        _videoKey.currentContext!.findRenderObject() as RenderRepaintBoundary;

        var image = await boundary.toImage(pixelRatio: 0.5);
        ByteData? byteData = await image.toByteData(format: ImageByteFormat.png);

        if (byteData != null) {
          Uint8List pngBytes = byteData.buffer.asUint8List();
          String base64Frame = base64Encode(pngBytes);
          _webSocket!.add(base64Frame); // Envia frame serializado via WebSocket
          print('Frame enviado via WebSocket.');
        }
      }

      Future.delayed(Duration(milliseconds: 100), _startFrameCapture);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('WebRTC App')),
      body: Center(
        child: RepaintBoundary(
          key: _videoKey,
          child: RTCVideoView(_localRenderer),
        ),
      ),
    );
  }
}
