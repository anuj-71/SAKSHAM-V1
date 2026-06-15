from src.session_manager import ConversationSession

s = ConversationSession()
s.add_message('Test', 'Hello')
print(f'Messages: {s.get_message_count()}, ExportStatus: "{s.last_export_status}"')
s.export()
print(f'After export - Status: "{s.last_export_status}", Path: "{s.last_export_path}"')
