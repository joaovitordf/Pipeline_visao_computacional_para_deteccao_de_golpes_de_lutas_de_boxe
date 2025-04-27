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
      title: 'Camera Boxe',
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final TextEditingController _ipController = TextEditingController(text: '192.168.1.100');
  final TextEditingController _portController = TextEditingController(text: '8765');

  bool _connected = false;

  RTCVideoRenderer _localRenderer = RTCVideoRenderer();
  MediaStream? _localStream;
  WebSocket? _webSocket;
  GlobalKey _videoKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    _inicializaRenderizador();
  }

  @override
  void dispose() {
    _ipController.dispose();
    _portController.dispose();
    _localRenderer.dispose();
    _localStream?.dispose();
    _webSocket?.close();
    super.dispose();
  }

  Future<void> _inicializaRenderizador() async {
    await _localRenderer.initialize();
  }

  Future<void> _conectaServidor(String ip, String port) async {
    try {
      _webSocket = await WebSocket.connect('ws://$ip:$port');
      print('Conectado ao servidor via WebSocket.');
      _webSocket!.listen((data) {
        print('Mensagem recebida: \$data');
      });
    } catch (e) {
      print('Erro ao conectar ao servidor via WebSocket: \$e');
      rethrow;
    }
  }

  Future<void> _inicializaRestricoes() async {
    final mediaConstraints = {
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
      print("Erro ao acessar a câmera: \$e");
    }
  }

  Future<void> _onClicarConectar() async {
    final ip = _ipController.text.trim();
    final port = _portController.text.trim();

    try {
      await _conectaServidor(ip, port);
      await _inicializaRestricoes();
      setState(() {
        _connected = true;
      });
    } catch (_) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Falha ao conectar. Verifique IP e porta.')),
      );
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
          _webSocket!.add(base64Frame);
          print('Frame enviado via WebSocket.');
        }
      }

      Future.delayed(Duration(milliseconds: 100), _startFrameCapture);
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_connected) {
      return Scaffold(
        appBar: AppBar(title: Text('Configuração de Conexão')),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              TextField(
                controller: _ipController,
                decoration: InputDecoration(labelText: 'IP do Servidor'),
              ),
              TextField(
                controller: _portController,
                decoration: InputDecoration(labelText: 'Porta'),
                keyboardType: TextInputType.number,
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: _onClicarConectar,
                child: Text('Conectar e Iniciar'),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: Text('Camera Boxe')),
      body: Center(
        child: RepaintBoundary(
          key: _videoKey,
          child: RTCVideoView(_localRenderer),
        ),
      ),
    );
  }
}
