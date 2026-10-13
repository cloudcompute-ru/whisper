import os
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class GatewayLaunchTest(unittest.TestCase):
    def launch_generated_app(self, environment):
        source = (ROOT / 'provision.sh').read_text().split("<<'PYAPP'\n", 1)[1].split('\nPYAPP', 1)[0]
        interface = Mock()
        gradio = types.SimpleNamespace(Interface=Mock(return_value=interface), Audio=Mock())
        whisper = types.SimpleNamespace(WhisperModel=Mock())
        with patch.dict(os.environ, environment, clear=True), patch.dict('sys.modules', {
            'gradio': gradio, 'faster_whisper': whisper,
        }):
            exec(compile(source, 'generated-app.py', 'exec'), {})
        return interface.launch.call_args.kwargs

    def test_default_launch_uses_authenticated_gateway_subpath(self):
        self.assertEqual({
            'server_name': '0.0.0.0', 'server_port': 7860, 'root_path': '/app-ui',
        }, self.launch_generated_app({}))

    def test_custom_gateway_path_and_port_are_supported(self):
        options = self.launch_generated_app({'GRADIO_ROOT_PATH': '/speech', 'WHISPER_PORT': '7861'})
        self.assertEqual('/speech', options['root_path'])
        self.assertEqual(7861, options['server_port'])

    def test_direct_hosting_can_explicitly_use_domain_root(self):
        self.assertEqual('', self.launch_generated_app({'GRADIO_ROOT_PATH': ''})['root_path'])


if __name__ == '__main__':
    unittest.main()
