# Configure a wandb workspace view for best showing logged gpt_batch_api data
# This code was written based on the source code of wandb-workspaces 0.1.11 (the library lost view information and didn't inherently support saving views to a new project, hence the need for manual low-level code)

# Imports
import copy
import json
import argparse
from typing import Any, Union, Optional
import wandb_workspaces.workspaces.internal

# Fake workspace view data class
class FakeWorkspaceViewspec:

	def __init__(self, data: Union[str, dict[str, Any]]):
		if isinstance(data, str):
			self.data = json.loads(data)
		elif isinstance(data, dict):
			self.data = copy.deepcopy(data)
		else:
			raise TypeError(f"Invalid data type: {type(data)}")

	# noinspection PyUnusedLocal
	def model_dump_json(self, **kwargs) -> str:
		return json.dumps(self.data, ensure_ascii=False, separators=(',', ':'))

	def __repr__(self) -> str:
		return 'WorkspaceViewspec(...)'

# Fake view class
class FakeView(wandb_workspaces.workspaces.internal.View):
	spec: FakeWorkspaceViewspec

# Main function
def main():

	parser = argparse.ArgumentParser(description="Configure a wandb workspace view for best showing logged gpt_batch_api data.", add_help=False)
	parser.add_argument('--help', '-h', action='help', default=argparse.SUPPRESS, help="Show this help message and exit")
	parser.add_argument('--src_entity', type=str, default='pallgeuer', help="Wandb entity to get source view from [default: %(default)s]")
	parser.add_argument('--src_project', type=str, default='gpt_batch_api_demo', help="Wandb project to get source view from [default: %(default)s]")
	parser.add_argument('--src_nw', type=str, default='u38mn3ltgfn', help="Wandb source view (alphanumeric string following '?nw=' in workspace URL when view is activated) to import and use [default: %(default)s]")
	parser.add_argument('--src_show', action='store_true', help="Show the source view contents")
	parser.add_argument('--dst_entity', type=str, default=None, help="Wandb entity to save the destination view to (a view is only saved if this is specified)")
	parser.add_argument('--dst_project', type=str, default='gpt_batch_api', help="Wandb project to save the destination view to [default: %(default)s]")
	parser.add_argument('--dst_nw', type=str, default=None, help="If specified, the wandb destination view to overwrite instead of creating a new view")
	parser.add_argument('--dst_show', action='store_true', help="Show the destination view contents")
	args = parser.parse_args()

	view = get_view(entity=args.src_entity, project=args.src_project, nw=args.src_nw)
	print(f"SOURCE VIEW: {view!r}")
	if args.src_show:
		print(json.dumps(view.spec.data, ensure_ascii=False, indent=2))
		print()

	if args.dst_entity is not None:
		view = save_view(view=view, entity=args.dst_entity, project=args.dst_project, nw=args.dst_nw)
		print(f"SAVED VIEW: {view!r}")
		if args.dst_show:
			print(json.dumps(view.spec.data, ensure_ascii=False, indent=2))
			print()
	else:
		print("NOT SAVING VIEW")

# Get an existing view from wandb
# noinspection PyProtectedMember
def get_view(entity: str, project: str, nw: str) -> FakeView:

	view_dict = wandb_workspaces.workspaces.internal.get_view_dict(entity=entity, project=project, view_name=nw)

	view = FakeView(
		entity=entity,
		project=project,
		display_name=view_dict['displayName'],
		name=wandb_workspaces.workspaces.internal._url_query_str_to_internal_name(name=nw),
		id=view_dict['id'],
		spec=FakeWorkspaceViewspec(data=view_dict['spec']),
	)

	view_spec_dump = view.spec.model_dump_json(by_alias=True, exclude_none=True)  # Should match what happens in wandb_workspaces.workspaces.internal.upsert_view2()
	if view_spec_dump != view_dict['spec']:
		print("[WARN] There is a difference between the received view spec and redumping it to str:")
		print(view_dict['spec'])
		print(view_spec_dump)

	return view

# Save a view to wandb (modifies the input view instance)
# noinspection PyProtectedMember
def save_view(view: FakeView, entity: str, project: str, nw: Optional[str]) -> FakeView:

	if nw is None:
		view.entity = entity
		view.project = project
		view.name = wandb_workspaces.workspaces.internal._generate_view_name()
		view.id = ''
		dst_view = view
	else:
		dst_view = get_view(entity=entity, project=project, nw=nw)
		dst_view.display_name = view.display_name
		dst_view.spec = view.spec

	data_section = dst_view.spec.data['section']
	data_section['customRunNames'].clear()
	data_section['customRunColors'].clear()
	for run_set in data_section['runSets']:
		run_set['selections']['tree'].clear()
	for panel_section in data_section['panelBankConfig']['sections']:
		for panel in panel_section['panels']:
			panel['config'].pop('overrideColors', None)
			panel['config'].pop('overrideLineWidths', None)

	response = wandb_workspaces.workspaces.internal.upsert_view2(view=dst_view)
	dst_view.name = response['upsertView']['view']['name']
	dst_view.id = response['upsertView']['view']['id']

	return dst_view

# Run main function
if __name__ == "__main__":
	main()
# EOF
