{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 5,
			"revision" : 4,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 124.0, 108.0, 1642.0, 948.0 ],
		"bglocked" : 0,
		"openinpresentation" : 1,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 1,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"format" : 6,
					"id" : "obj-4806",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 776.809526681900024, 2927.285707235336304, 50.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.5, 151.0, 50.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4807",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 2232.809526681900024, 3853.285707235336304, 135.0, 22.0 ],
					"text" : "r ripple_phase_duration"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4808",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 2092.809526681900024, 3853.285707235336304, 114.0, 22.0 ],
					"text" : "r ripple_phase_loop"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4809",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 409.809526681900024, 3780.285707235336304, 117.0, 22.0 ],
					"text" : "r ripple_amp_z_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4811",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 279.809526681900024, 3780.285707235336304, 114.0, 22.0 ],
					"text" : "r ripple_amp_z_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4813",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 173.809526681900024, 3780.285707235336304, 88.0, 22.0 ],
					"text" : "r ripple_amp_z"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4815",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 409.809526681900024, 3753.285707235336304, 117.0, 22.0 ],
					"text" : "r ripple_amp_y_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4816",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 279.809526681900024, 3753.285707235336304, 114.0, 22.0 ],
					"text" : "r ripple_amp_y_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4817",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 173.809526681900024, 3753.285707235336304, 88.0, 22.0 ],
					"text" : "r ripple_amp_y"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4818",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 409.809526681900024, 3720.285707235336304, 117.0, 22.0 ],
					"text" : "r ripple_amp_x_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4819",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 279.809526681900024, 3720.285707235336304, 114.0, 22.0 ],
					"text" : "r ripple_amp_x_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4820",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 173.809526681900024, 3720.285707235336304, 88.0, 22.0 ],
					"text" : "r ripple_amp_x"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4821",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 409.809526681900024, 3877.285707235336304, 115.0, 22.0 ],
					"text" : "r ripple_freq_z_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4822",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 279.809526681900024, 3877.285707235336304, 111.0, 22.0 ],
					"text" : "r ripple_freq_z_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4823",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 173.809526681900024, 3877.285707235336304, 85.0, 22.0 ],
					"text" : "r ripple_freq_z"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4824",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 409.809526681900024, 3847.285707235336304, 115.0, 22.0 ],
					"text" : "r ripple_freq_y_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4825",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 279.809526681900024, 3847.285707235336304, 111.0, 22.0 ],
					"text" : "r ripple_freq_y_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4826",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 173.809526681900024, 3847.285707235336304, 85.0, 22.0 ],
					"text" : "r ripple_freq_y"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4827",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 409.809526681900024, 3814.285707235336304, 115.0, 22.0 ],
					"text" : "r ripple_freq_x_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4828",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 279.809526681900024, 3814.285707235336304, 111.0, 22.0 ],
					"text" : "r ripple_freq_x_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4829",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 173.809526681900024, 3814.285707235336304, 85.0, 22.0 ],
					"text" : "r ripple_freq_x"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4830",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 412.809526681900024, 3969.285707235336304, 127.0, 22.0 ],
					"text" : "r ripple_phase_z_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4831",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 282.809526681900024, 3969.285707235336304, 123.0, 22.0 ],
					"text" : "r ripple_phase_z_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4832",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 176.809526681900024, 3969.285707235336304, 97.0, 22.0 ],
					"text" : "r ripple_phase_z"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4833",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 412.809526681900024, 3939.285707235336304, 127.0, 22.0 ],
					"text" : "r ripple_phase_y_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4834",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 282.809526681900024, 3939.285707235336304, 123.0, 22.0 ],
					"text" : "r ripple_phase_y_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4835",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 176.809526681900024, 3939.285707235336304, 97.0, 22.0 ],
					"text" : "r ripple_phase_y"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4836",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 412.809526681900024, 3909.285707235336304, 127.0, 22.0 ],
					"text" : "r ripple_phase_x_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4837",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 282.809526681900024, 3909.285707235336304, 123.0, 22.0 ],
					"text" : "r ripple_phase_x_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4838",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 176.809526681900024, 3909.285707235336304, 97.0, 22.0 ],
					"text" : "r ripple_phase_x"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4839",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 423.809526681900024, 3494.285707235336304, 138.0, 22.0 ],
					"text" : "s ambient_scale_max_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4840",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 285.809526681900024, 3494.285707235336304, 135.0, 22.0 ],
					"text" : "s ambient_scale_min_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4841",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 158.809526681900024, 3494.285707235336304, 109.0, 22.0 ],
					"text" : "s ambient_scale_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4842",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 423.809526681900024, 3526.285707235336304, 140.0, 22.0 ],
					"text" : "s diffusion_scale_max_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4843",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 285.809526681900024, 3526.285707235336304, 136.0, 22.0 ],
					"text" : "s diffusion_scale_min_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4844",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 158.809526681900024, 3526.285707235336304, 110.0, 22.0 ],
					"text" : "s diffusion_scale_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4845",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 423.809526681900024, 3556.285707235336304, 141.0, 22.0 ],
					"text" : "s specular_scale_max_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4846",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 285.809526681900024, 3556.285707235336304, 137.0, 22.0 ],
					"text" : "s specular_scale_min_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4847",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 158.809526681900024, 3556.285707235336304, 111.0, 22.0 ],
					"text" : "s specular_scale_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4848",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 423.809526681900024, 3586.285707235336304, 135.0, 22.0 ],
					"text" : "s specular_pow_max_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4849",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 285.809526681900024, 3586.285707235336304, 131.0, 22.0 ],
					"text" : "s specular_pow_min_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4850",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 158.809526681900024, 3586.285707235336304, 105.0, 22.0 ],
					"text" : "s specular_pow_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4851",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 423.809526681900024, 3616.285707235336304, 130.0, 22.0 ],
					"text" : "r occlusion_scale_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4852",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 285.809526681900024, 3616.285707235336304, 142.0, 22.0 ],
					"text" : "s occlusion_scale_min_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4853",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 158.809526681900024, 3616.285707235336304, 116.0, 22.0 ],
					"text" : "s occlusion_scale_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4854",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 423.809526681900024, 3644.285707235336304, 133.0, 22.0 ],
					"text" : "r occlusion_range_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4855",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 285.809526681900024, 3644.285707235336304, 129.0, 22.0 ],
					"text" : "r occlusion_range_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4856",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 158.809526681900024, 3644.285707235336304, 103.0, 22.0 ],
					"text" : "r occlusion_range"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4857",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 409.809526681900024, 3676.285707235336304, 119.0, 22.0 ],
					"text" : "r occlusion_res_max"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4858",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 279.809526681900024, 3676.285707235336304, 115.0, 22.0 ],
					"text" : "r occlusion_res_min"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4859",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 173.809526681900024, 3676.285707235336304, 89.0, 22.0 ],
					"text" : "r occlusion_res"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4860",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2053.809526681900024, 3349.285707235336304, 71.0, 22.0 ],
					"text" : "s color_b_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4861",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2053.809526681900024, 3323.285707235336304, 71.0, 22.0 ],
					"text" : "s color_g_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4862",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2053.809526681900024, 3299.285707235336304, 68.0, 22.0 ],
					"text" : "s color_r_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4863",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2053.809526681900024, 3437.285707235336304, 127.0, 22.0 ],
					"text" : "s occlusion_color_b_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4864",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2053.809526681900024, 3411.285707235336304, 127.0, 22.0 ],
					"text" : "s occlusion_color_g_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4865",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2053.809526681900024, 3380.285707235336304, 125.0, 22.0 ],
					"text" : "s occlusion_color_r_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4884",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2128.809526681900024, 3030.285707235336304, 95.0, 22.0 ],
					"text" : "s size_z_max_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4885",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2047.809526681900024, 3030.285707235336304, 65.0, 22.0 ],
					"text" : "s size_z_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4886",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2128.809526681900024, 2997.285707235336304, 95.0, 22.0 ],
					"text" : "s size_y_max_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4887",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2047.809526681900024, 2997.285707235336304, 65.0, 22.0 ],
					"text" : "s size_y_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4888",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2128.809526681900024, 2965.285707235336304, 95.0, 22.0 ],
					"text" : "s size_x_max_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4889",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2047.809526681900024, 2965.285707235336304, 65.0, 22.0 ],
					"text" : "s size_x_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4890",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2128.809526681900024, 3056.285707235336304, 91.0, 22.0 ],
					"text" : "s round_max_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4891",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2047.809526681900024, 3056.285707235336304, 62.0, 22.0 ],
					"text" : "s round_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4892",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2122.809526681900024, 3086.285707235336304, 100.0, 22.0 ],
					"text" : "s smooth_max_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4893",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2047.809526681900024, 3086.285707235336304, 71.0, 22.0 ],
					"text" : "s smooth_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4894",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2047.809526681900024, 2927.285707235336304, 76.0, 22.0 ],
					"text" : "s primitive_0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4897",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 885.809526681900024, 2929.285707235336304, 29.5, 22.0 ],
					"text" : "+ 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4898",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 2346.809526681900024, 4031.285707235336304, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4899",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 2200.809526681900024, 3929.285707235336304, 90.0, 22.0 ],
					"text" : "expr $f1* 1000."
				}

			}
, 			{
				"box" : 				{
					"format" : 6,
					"id" : "obj-4900",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 2200.809526681900024, 4066.285707235336304, 50.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4901",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 2200.809526681900024, 4031.285707235336304, 92.0, 22.0 ],
					"text" : "expr $f1 / 1000."
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4902",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2131.809526681900024, 3899.285707235336304, 61.818184971809387, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 830.894736230373383, 808.088496327400208, 61.818184971809387, 20.0 ],
					"text" : "duration"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4903",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2034.809526681900024, 3901.285707235336304, 48.181821823120117, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 732.98560905456543, 810.210527658462524, 48.181821823120117, 20.0 ],
					"text" : "loop"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4904",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 2092.809526681900024, 3947.285707235336304, 32.0, 22.0 ],
					"text" : "gate"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4905",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 2092.809526681900024, 3897.285707235336304, 24.0, 24.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 791.167425155639648, 806.210527658462524, 24.0, 24.0 ],
					"varname" : "object_phaseLoop[11]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4906",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 2200.809526681900024, 3899.285707235336304, 50.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 899.167477369308472, 808.210527658462524, 50.0, 22.0 ],
					"varname" : "object_phaseLoop_duration[11]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4907",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 2200.809526681900024, 3960.285707235336304, 105.0, 22.0 ],
					"text" : "sprintf 0\\, 1000 %d"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-4908",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"patching_rect" : [ 2200.809526681900024, 3990.285707235336304, 166.0, 23.0 ],
					"text" : "line 1 20"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4909",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3965.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4910",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3965.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4911",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3965.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4912",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3962.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 827.40328574180603, 50.0, 22.0 ],
					"varname" : "objectPhaseZ_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4913",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3962.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 827.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectPhaseZ_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4914",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3962.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 827.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectPhaseZ[10]"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4915",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3965.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 828.40328574180603, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4916",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3935.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4917",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3935.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4918",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3935.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4919",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3932.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 804.40328574180603, 50.0, 22.0 ],
					"varname" : "objectPhaseY_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4920",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3932.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 804.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectPhaseY_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4921",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3932.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 804.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectPhaseY[10]"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4922",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3935.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 805.40328574180603, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4923",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3905.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4924",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3905.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4925",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3905.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4926",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3902.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 781.40328574180603, 50.0, 22.0 ],
					"varname" : "objectPhaseX_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4927",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3902.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 781.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectPhaseX_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4928",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3902.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 781.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectPhaseX[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4929",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1932.809526681900024, 3900.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4930",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3900.285707235336304, 145.0, 22.0 ],
					"text" : "/vis/objectphase $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4931",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1591.809526681900024, 3905.285707235336304, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4932",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3902.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 735.394750356674194, 781.40328574180603, 20.0, 20.0 ],
					"varname" : "objectPhase_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4933",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3902.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 764.394750356674194, 781.40328574180603, 178.0, 20.0 ],
					"text" : "Object Phase"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4934",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3905.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 782.40328574180603, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4935",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3873.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4936",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3873.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4937",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3873.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4938",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3870.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 758.40328574180603, 50.0, 22.0 ],
					"varname" : "objectFrequencyZ_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4939",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3870.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 758.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectFrequencyZ_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4940",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3870.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 758.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectFrequencyZ[10]"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4941",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3873.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 759.40328574180603, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4942",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3843.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4943",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3843.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4944",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3843.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4945",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3840.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 735.40328574180603, 50.0, 22.0 ],
					"varname" : "objectFrequencyY_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4946",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3840.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 735.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectFrequencyY_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4947",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3840.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 735.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectFrequencyY[10]"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4948",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3843.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 736.40328574180603, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4949",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3813.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4950",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3813.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4951",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3813.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4952",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3810.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 712.40328574180603, 50.0, 22.0 ],
					"varname" : "objectFrequencyX_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4953",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3810.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 712.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectFrequencyX_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4954",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3810.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 712.40328574180603, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectFrequencyX[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4955",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1932.809526681900024, 3808.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4956",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3808.285707235336304, 165.0, 22.0 ],
					"text" : "/vis/objectfrequency $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4957",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1591.809526681900024, 3813.285707235336304, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4958",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3810.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 735.394750356674194, 712.40328574180603, 20.0, 20.0 ],
					"varname" : "objectFrequency_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4959",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3810.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 764.394750356674194, 712.40328574180603, 178.0, 20.0 ],
					"text" : "Object Frequency"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4960",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3813.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 713.40328574180603, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4961",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3783.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4962",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3783.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4963",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3783.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4964",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3780.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1056.394750356674194, 686.210527658462524, 50.0, 22.0 ],
					"varname" : "objectAmplitudeZ_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4965",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3780.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1005.394750356674194, 686.210527658462524, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectAmplitudeZ_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4966",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3780.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 953.394750356674194, 686.210527658462524, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectAmplitudeZ[10]"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4967",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3783.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1108.394750356674194, 687.210527658462524, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundPosZ[22]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4968",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3753.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4969",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3753.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4970",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3753.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4971",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3750.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1056.394750356674194, 663.210527658462524, 50.0, 22.0 ],
					"varname" : "objectAmplitudeY_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4972",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3750.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1005.394750356674194, 663.210527658462524, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectAmplitudeY_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4973",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3750.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 953.394750356674194, 663.210527658462524, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectAmplitudeY[10]"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4974",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3753.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1108.394750356674194, 664.210527658462524, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4975",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3723.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4976",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3723.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4977",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3723.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4978",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3720.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1056.394750356674194, 640.210527658462524, 50.0, 22.0 ],
					"varname" : "objectAmplitudeX_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4979",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3720.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1005.394750356674194, 640.210527658462524, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectAmplitudeX_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4980",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3720.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 953.394750356674194, 640.210527658462524, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectAmplitudeX[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4981",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1932.809526681900024, 3718.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4982",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3718.285707235336304, 164.0, 22.0 ],
					"text" : "/vis/objectamplitude $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4983",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1591.809526681900024, 3723.285707235336304, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4984",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3720.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 736.394750356674194, 640.210527658462524, 20.0, 20.0 ],
					"varname" : "objectAmplitude_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4985",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3720.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 765.394750356674194, 640.210527658462524, 178.0, 20.0 ],
					"text" : "Object Amplitude"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4986",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3723.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1108.394750356674194, 641.210527658462524, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4987",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3589.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4988",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3589.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4989",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3589.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4990",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3587.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 521.602875709533691, 50.0, 22.0 ],
					"varname" : "objectSpecularPow_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4991",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3587.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 521.602875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectSpecularPow_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4992",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3587.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 521.602875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectSpecularPow[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4993",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2130.809526681900024, 3584.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4994",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3584.285707235336304, 146.0, 22.0 ],
					"text" : "/vis/objectspecularpow $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4995",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3587.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 736.394750356674194, 521.602875709533691, 20.0, 20.0 ],
					"varname" : "objectSpecularPow_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4996",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3587.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 765.394750356674194, 521.602875709533691, 178.0, 20.0 ],
					"text" : "Object Specular Pow"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4997",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3589.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 522.602875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4998",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3559.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4999",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3559.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5000",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3559.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5001",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3557.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 496.602875709533691, 50.0, 22.0 ],
					"varname" : "objectSpecularScale_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5002",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3557.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 496.602875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectSpecularScale_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5003",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3557.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 496.602875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectSpecularScale[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5004",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2130.809526681900024, 3554.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5005",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3554.285707235336304, 152.0, 22.0 ],
					"text" : "/vis/objectspecularscale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5006",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3557.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 736.394750356674194, 496.602875709533691, 20.0, 20.0 ],
					"varname" : "objectSpecularScale_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5007",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3557.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 765.394750356674194, 496.602875709533691, 178.0, 20.0 ],
					"text" : "ObjectSpecular Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5008",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3559.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 497.602875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5009",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3529.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5010",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3529.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5011",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3529.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5012",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3527.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 475.102875709533691, 50.0, 22.0 ],
					"varname" : "objectDiffuseScale_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5013",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3527.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 475.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectDiffuseScale_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5014",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3527.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 475.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectDiffuseScale[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5015",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2130.809526681900024, 3524.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5016",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3524.285707235336304, 142.0, 22.0 ],
					"text" : "/vis/objectdiffusescale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5017",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3527.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 736.394750356674194, 475.102875709533691, 20.0, 20.0 ],
					"varname" : "objectDiffuseScale_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5018",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3527.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 765.394750356674194, 475.102875709533691, 178.0, 20.0 ],
					"text" : "Object Diffusion Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5019",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3529.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 476.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5020",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3499.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5021",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3499.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5022",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3499.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5023",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3497.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 450.102875709533691, 50.0, 22.0 ],
					"varname" : "objectAmbientScale_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5024",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3497.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 450.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectAmbientScale_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5025",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3497.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 450.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectAmbientScale[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5026",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2130.809526681900024, 3494.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5027",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3494.285707235336304, 149.0, 22.0 ],
					"text" : "/vis/objectambientscale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5028",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3497.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 736.394750356674194, 450.102875709533691, 20.0, 20.0 ],
					"varname" : "objectAmbientScale_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5029",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3497.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 765.394750356674194, 450.102875709533691, 178.0, 20.0 ],
					"text" : "Object Ambient Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5030",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3499.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 451.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5031",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3619.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5032",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3619.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5033",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3619.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5034",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3617.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 546.102875709533691, 50.0, 22.0 ],
					"varname" : "objectOcclusionScale_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5035",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3617.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 546.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectOcclusionScale_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5036",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3617.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 546.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectOcclusionScale[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5037",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2130.809526681900024, 3614.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5038",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3614.285707235336304, 157.0, 22.0 ],
					"text" : "/vis/objectocclusionscale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5039",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3617.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 736.394750356674194, 546.102875709533691, 20.0, 20.0 ],
					"varname" : "objectOcclusionScale_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5040",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3617.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 765.394750356674194, 546.102875709533691, 178.0, 20.0 ],
					"text" : "Object Occlusion Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5041",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3619.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 547.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5042",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3679.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5043",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3679.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5044",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3679.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5045",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3677.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 592.102875709533691, 50.0, 22.0 ],
					"varname" : "objectOcclusionResol_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5046",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3677.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 592.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectOcclusionResol_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5047",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3677.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 592.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectOcclusionResol[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5048",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2130.809526681900024, 3674.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5049",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3674.285707235336304, 181.0, 22.0 ],
					"text" : "/vis/objectocclusionresolution $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5050",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3677.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 736.394750356674194, 592.102875709533691, 20.0, 20.0 ],
					"varname" : "objectOcclusionResol_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5051",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3677.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 765.394750356674194, 592.102875709533691, 178.0, 20.0 ],
					"text" : "Object Occlusion Resolution"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5052",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3679.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 593.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5053",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3649.285707235336304, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5054",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3649.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5055",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3649.285707235336304, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5056",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3647.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1055.394750356674194, 569.102875709533691, 50.0, 22.0 ],
					"varname" : "objectOcclusionRange_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5057",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 829.809526681900024, 3647.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1004.394750356674194, 569.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectOcclusionRange_min[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5058",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3647.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 569.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectOcclusionRange[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5059",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 2130.809526681900024, 3644.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5060",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3644.285707235336304, 159.0, 22.0 ],
					"text" : "/vis/objectocclusionrange $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5061",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3647.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 736.394750356674194, 569.102875709533691, 20.0, 20.0 ],
					"varname" : "objectOcclusionRange_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5062",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3647.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 765.394750356674194, 569.102875709533691, 178.0, 20.0 ],
					"text" : "Object Occlusion Range"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5063",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3649.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 570.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5064",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1277.809526681900024, 3469.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5065",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3460.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 427.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectOcclusionColorB[10]"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5066",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3469.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 428.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5067",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1277.809526681900024, 3437.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5068",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3428.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 404.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectOcclusionColorG[10]"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5069",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3437.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 405.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5070",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1277.809526681900024, 3407.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5071",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3398.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 381.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectOcclusionColorR[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5072",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1631.809526681900024, 3411.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5073",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1434.809526681900024, 3411.285707235336304, 188.0, 22.0 ],
					"text" : "/vis/objectocclusioncolor $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5074",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1332.809526681900024, 3411.285707235336304, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5075",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3398.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 735.394750356674194, 381.102875709533691, 20.0, 20.0 ],
					"varname" : "objectOcclusionColor_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5076",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3398.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 764.394750356674194, 381.102875709533691, 178.0, 20.0 ],
					"text" : "Object Occlusion Color"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5077",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3407.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 382.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5078",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1277.809526681900024, 3372.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5079",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3363.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 360.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectColorB[10]"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5080",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3372.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 361.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5081",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1277.809526681900024, 3340.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5082",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3331.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 337.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectColorG[10]"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5083",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3340.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 338.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5084",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1277.809526681900024, 3310.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5085",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3301.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 952.394750356674194, 314.102875709533691, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "objectColorR[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5086",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1621.546368837356567, 3314.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5087",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1434.809526681900024, 3314.285707235336304, 138.0, 22.0 ],
					"text" : "/vis/objectcolor $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5088",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1332.809526681900024, 3314.285707235336304, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5089",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3301.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 735.394750356674194, 314.102875709533691, 20.0, 20.0 ],
					"varname" : "objectColor_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5090",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3301.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 764.394750356674194, 314.102875709533691, 178.0, 20.0 ],
					"text" : "Object Color"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5091",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3310.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1107.394750356674194, 315.102875709533691, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5099",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3091.285707235336304, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5100",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3091.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5101",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3091.285707235336304, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5102",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3088.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1054.5, 267.68182373046875, 50.0, 22.0 ],
					"varname" : "groundSmooth_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5103",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3088.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 951.5, 267.68182373046875, 50.0, 22.0 ],
					"triangle" : 0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5104",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1932.809526681900024, 3086.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5105",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3086.285707235336304, 118.0, 22.0 ],
					"text" : "/vis/objectsmooth $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5106",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3088.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 735.5, 267.68182373046875, 20.0, 20.0 ],
					"varname" : "groundSmooth_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5107",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3088.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 764.5, 267.68182373046875, 178.0, 20.0 ],
					"text" : "Object Smooth"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5108",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3091.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1106.5, 268.68182373046875, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundSmooth[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5109",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3061.285707235336304, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5110",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3061.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5111",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3061.285707235336304, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5112",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3058.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1054.5, 244.68182373046875, 50.0, 22.0 ],
					"varname" : "groundRound_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5113",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3058.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 951.5, 244.68182373046875, 50.0, 22.0 ],
					"triangle" : 0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5114",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1932.809526681900024, 3056.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5115",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 3056.285707235336304, 109.0, 22.0 ],
					"text" : "/vis/objectround $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5116",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 3058.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 735.5, 244.68182373046875, 20.0, 20.0 ],
					"varname" : "groundRound_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5117",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 3058.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 764.5, 244.68182373046875, 178.0, 20.0 ],
					"text" : "Object Round"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5118",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3061.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1106.5, 245.68182373046875, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundRound[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5119",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3030.285707235336304, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5120",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3030.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5121",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3030.285707235336304, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5122",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 3027.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1054.5, 222.0, 50.0, 22.0 ],
					"varname" : "groundSizeZ_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5123",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 3027.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 951.5, 222.0, 50.0, 22.0 ],
					"triangle" : 0
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5124",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3030.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1106.5, 223.0, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundSizeZ[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5125",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 3000.285707235336304, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5126",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 3000.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5127",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 3000.285707235336304, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5128",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 2997.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1054.5, 199.0, 50.0, 22.0 ],
					"varname" : "groundSizeY_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5129",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 2997.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 951.5, 199.0, 50.0, 22.0 ],
					"triangle" : 0
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5130",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 3000.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1106.5, 200.0, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundSizeY[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5131",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1590.809526681900024, 2970.285707235336304, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5132",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1399.809526681900024, 2970.285707235336304, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5133",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 2970.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5134",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1271.809526681900024, 2970.285707235336304, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5135",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 881.809526681900024, 2967.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1054.5, 176.0, 50.0, 22.0 ],
					"varname" : "groundSizeX_max[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-5136",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 778.809526681900024, 2967.285707235336304, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 951.5, 176.0, 50.0, 22.0 ],
					"triangle" : 0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5137",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1932.809526681900024, 2965.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5138",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 2965.285707235336304, 133.0, 22.0 ],
					"text" : "/vis/objectsize $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5139",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 2967.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 735.5, 176.0, 20.0, 20.0 ],
					"varname" : "groundSize_interest[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5140",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 2967.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 764.5, 176.0, 178.0, 20.0 ],
					"text" : "Object Size"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5141",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 2970.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1106.5, 177.0, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundSizeX[10]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5142",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1533.809526681900024, 2932.285707235336304, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5143",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1932.809526681900024, 2927.285707235336304, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5144",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1758.809526681900024, 2927.285707235336304, 123.0, 22.0 ],
					"text" : "/vis/objectprimitive $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5145",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.809526681900024, 2929.285707235336304, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 735.5, 151.0, 20.0, 20.0 ],
					"varname" : "groundPrimitive_interest[11]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-5146",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 575.809526681900024, 2929.285707235336304, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 764.5, 151.0, 178.0, 20.0 ],
					"text" : "Object Primitive"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-5147",
					"maxclass" : "slider",
					"min" : -1.0,
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 939.809526681900024, 2932.285707235336304, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1106.5, 152.0, 320.0, 20.0 ],
					"size" : 19.0,
					"varname" : "groundPrimitive[11]"
				}

			}
, 			{
				"box" : 				{
					"format" : 6,
					"id" : "obj-4812",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2615.0, 50.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 1022.903620958328247, 50.0, 22.0 ],
					"varname" : "edgePrimitive"
				}

			}
, 			{
				"box" : 				{
					"format" : 6,
					"id" : "obj-4810",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1999.0, 50.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 851.903620958328247, 50.0, 22.0 ],
					"varname" : "jointPrimitive"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4783",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 1275.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4784",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 1275.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4785",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 1275.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4786",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 1272.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.659354865550995, 1906.593499779701233, 50.0, 22.0 ],
					"varname" : "fogMaxDist_max[1]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4787",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 1272.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 296.56045126914978, 1906.593499779701233, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "fogMaxDist_min[1]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4788",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 1272.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.362646520137787, 1906.593499779701233, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "fogMaxDist[1]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4789",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.25, 1270.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4790",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 1270.0, 97.0, 22.0 ],
					"text" : "/vis/raywiggle $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4791",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.25, 1272.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.97802060842514, 1906.593499779701233, 20.0, 20.0 ],
					"varname" : "fogMaxDist_interest[1]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4792",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.25, 1272.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1906.593499779701233, 178.0, 20.0 ],
					"text" : "Ray Wiggle"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4793",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 1275.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.857159614562988, 1907.692400932312012, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4794",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 1241.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4795",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 1241.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4796",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 1241.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4797",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 1238.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.659354865550995, 1884.615476727485657, 50.0, 22.0 ],
					"varname" : "fogMinDist_max[1]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4798",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 1238.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 296.56045126914978, 1884.615476727485657, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "fogMinDist_min[1]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4799",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 1238.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.362646520137787, 1884.615476727485657, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "fogMinDist[1]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4800",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.25, 1236.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4801",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 1236.0, 103.0, 22.0 ],
					"text" : "/vis/rayrotation $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4802",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.25, 1238.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.97802060842514, 1884.615476727485657, 20.0, 20.0 ],
					"varname" : "fogMinDist_interest[1]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4803",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.25, 1238.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1884.615476727485657, 178.0, 20.0 ],
					"text" : "Ray Rotation"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4804",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 1241.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.857159614562988, 1886.263828456401825, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4739",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 1164.5, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4740",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 1164.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4741",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 1164.5, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4742",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 1162.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 634.81817626953125, 50.0, 22.0 ],
					"varname" : "fogMaxDist_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4743",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 1162.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 634.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "fogMaxDist_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4744",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 1162.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 634.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "fogMaxDist"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4745",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.25, 1159.5, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4746",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 1159.5, 105.0, 22.0 ],
					"text" : "/vis/fogmaxdist $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4747",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1162.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 634.81817626953125, 20.0, 20.0 ],
					"varname" : "fogMaxDist_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4748",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1162.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 634.81817626953125, 178.0, 20.0 ],
					"text" : "Fog Max Dist"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4749",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 1164.5, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 635.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4750",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 1130.5, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4751",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 1130.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4752",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 1130.5, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4753",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 1128.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 612.81817626953125, 50.0, 22.0 ],
					"varname" : "fogMinDist_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4754",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 1128.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 612.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "fogMinDist_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4755",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 1128.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 612.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "fogMinDist"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4756",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.25, 1125.5, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4757",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 1125.5, 101.0, 22.0 ],
					"text" : "/vis/fogmindist $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4758",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1128.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 612.81817626953125, 20.0, 20.0 ],
					"varname" : "fogMinDist_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4759",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1128.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 612.81817626953125, 178.0, 20.0 ],
					"text" : "Fog Min Dist"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4760",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 1130.5, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 613.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4761",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 1095.5, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4762",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 1095.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4763",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 1095.5, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4764",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 1093.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 566.81817626953125, 50.0, 22.0 ],
					"varname" : "shadowStrength_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4765",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 1093.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 566.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "shadowStrength_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4766",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 1093.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 566.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "shadowStrength"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4767",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.25, 1090.5, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4768",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 1090.5, 131.0, 22.0 ],
					"text" : "/vis/shadowstrength $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4769",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1093.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 566.81817626953125, 20.0, 20.0 ],
					"varname" : "shadowStrength_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4770",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1093.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 566.81817626953125, 178.0, 20.0 ],
					"text" : "Shadow Strength"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4771",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 1095.5, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 567.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4772",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 1061.5, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4773",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 1061.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4774",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 1061.5, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4775",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 1059.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 590.81817626953125, 50.0, 22.0 ],
					"varname" : "shadowSmooth_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4776",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 1059.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 590.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "shadowSmooth_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4777",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 1059.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 590.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "shadowSmooth"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4778",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.25, 1056.5, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4779",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 1056.5, 127.0, 22.0 ],
					"text" : "/vis/shadowsmooth $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4780",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1059.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 590.81817626953125, 20.0, 20.0 ],
					"varname" : "shadowSmooth_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4781",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1059.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 590.81817626953125, 178.0, 20.0 ],
					"text" : "Shadow Smooth"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4782",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 1061.5, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 591.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-304",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1112.25, 1008.755615234375, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-919",
					"maxclass" : "newobj",
					"numinlets" : 4,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1077.0, 1016.447265625, 114.0, 22.0 ],
					"text" : "pack 0.1 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4685",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "float", "float", "float", "float" ],
					"patching_rect" : [ 1077.25, 973.447265625, 127.0, 22.0 ],
					"text" : "unpack 0.1 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4686",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "list", "" ],
					"patching_rect" : [ 1077.25, 938.947265625, 77.0, 22.0 ],
					"text" : "jit.euler2quat"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4687",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 968.947265625, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4688",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 968.947265625, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4689",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 968.947265625, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4690",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 966.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 333.085444688796997, 50.0, 22.0 ],
					"varname" : "sceneRotZ_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4691",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 966.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 333.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "sceneRotZ_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4692",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 966.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 333.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "sceneRotZ"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4693",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 968.947265625, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 334.085444688796997, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundRotZ[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4694",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 938.947265625, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4695",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 938.947265625, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4696",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 938.947265625, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4697",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 936.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 310.085444688796997, 50.0, 22.0 ],
					"varname" : "sceneRotY_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4698",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 936.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 310.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "sceneRot_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4699",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 936.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 310.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "sceneRotY"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4700",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 938.947265625, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 311.085444688796997, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundRotY[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4701",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 908.947265625, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4702",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 908.947265625, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4703",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 908.947265625, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4704",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 906.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 287.085444688796997, 50.0, 22.0 ],
					"varname" : "sceneRotX_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4705",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 906.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 287.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "sceneRotX_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4706",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 906.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 287.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "sceneRotX"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4707",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1444.25, 903.947265625, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4708",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 903.947265625, 169.0, 22.0 ],
					"text" : "/vis/scenerotation $1 $2 $3 $4"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4709",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1077.25, 908.947265625, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4710",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 23.0, 906.447265625, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 287.085444688796997, 20.0, 20.0 ],
					"varname" : "sceneRot_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4711",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 906.447265625, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 287.085444688796997, 178.0, 20.0 ],
					"text" : "Scene Rotation"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4712",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 908.947265625, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 288.085444688796997, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundRotX[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4713",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 876.947265625, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4714",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 876.947265625, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4715",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 876.947265625, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4716",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 874.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 258.085444688796997, 50.0, 22.0 ],
					"varname" : "scenePositionZ_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4717",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 874.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 258.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "scenePositionZ_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4718",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 874.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 258.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "scenePositionZ"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4719",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 876.947265625, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 259.085444688796997, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundPosZ[21]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4720",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 846.947265625, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4721",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 846.947265625, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4722",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 846.947265625, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4723",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 844.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 235.085444688796997, 50.0, 22.0 ],
					"varname" : "scenePositionY_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4724",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 844.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 235.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "scenePositionY_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4725",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 844.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 235.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "scenePositionY"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4726",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 846.947265625, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 236.085444688796997, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "groundPosY[10]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4727",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 816.947265625, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4728",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 816.947265625, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-4729",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 816.947265625, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4730",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 814.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 212.085444688796997, 50.0, 22.0 ],
					"varname" : "scenePositionX_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4731",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 814.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 212.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "scenePositionX_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-4732",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 814.447265625, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 212.085444688796997, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "scenePositionX"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4733",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.25, 811.947265625, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4734",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 811.947265625, 153.0, 22.0 ],
					"text" : "/vis/sceneposition $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4735",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1077.25, 816.947265625, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4736",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 23.0, 814.447265625, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 212.085444688796997, 20.0, 20.0 ],
					"varname" : "scenePos_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-4737",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 814.447265625, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 212.085444688796997, 178.0, 20.0 ],
					"text" : "Scene Position"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-4738",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 816.947265625, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 213.085444688796997, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1004",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1143.25, 442.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-1003",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 442.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 662.18182373046875, 178.0, 20.0 ],
					"text" : "Background Active"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1002",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 260.25, 442.0, 24.0, 24.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 660.18182373046875, 24.0, 24.0 ],
					"varname" : "bg_active"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-579",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 315.25, 442.0, 80.0, 22.0 ],
					"text" : "/vis/bg_on $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-921",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 367.0, 2618.0, 29.5, 22.0 ],
					"text" : "+ 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-920",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 338.0, 1999.0, 29.5, 22.0 ],
					"text" : "+ 1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-1067",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 889.0, 2866.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1068",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1023.0, 2866.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-1069",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 761.0, 2866.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-1070",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 371.0, 2863.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1054.394750356674194, 865.377307653427124, 50.0, 22.0 ],
					"varname" : "jointEdgeSmooth_max[1]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-1071",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 268.0, 2863.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 951.394750356674194, 865.377307653427124, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointEdgeSmooth[1]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1072",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1422.0, 2861.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1073",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1248.0, 2861.0, 139.0, 22.0 ],
					"text" : "/vis/skelobjectsmooth $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1074",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 35.0, 2863.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 735.394750356674194, 865.377307653427124, 20.0, 20.0 ],
					"varname" : "jointEdgeSmooth_interest[1]"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-1075",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 65.0, 2863.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 764.394750356674194, 865.377307653427124, 178.0, 20.0 ],
					"text" : "Skel Object Smooth"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-1076",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 429.0, 2866.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 1106.394750356674194, 866.377307653427124, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-658",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2498.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-659",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2498.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-660",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2498.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-661",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2495.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1760.5, 50.0, 22.0 ],
					"varname" : "edgeSpecularPow_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-662",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 2495.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1760.5, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeSpecularPow_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-663",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2495.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1760.5, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeSpecularPow"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-664",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 2493.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-665",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2493.0, 141.0, 22.0 ],
					"text" : "/vis/edgespecularpow $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-666",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2495.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1760.5, 20.0, 20.0 ],
					"varname" : "edgeSpecularPow_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-667",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2495.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1760.5, 178.0, 20.0 ],
					"text" : "Edge Specular Pow"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-668",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2498.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1761.5, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-669",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2468.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-670",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2468.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-671",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2468.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-672",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2465.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1735.5, 50.0, 22.0 ],
					"varname" : "edgeSpecularScale_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-673",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 2465.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1735.5, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeSpecularScale_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-674",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2465.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1735.5, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeSpecularScale"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-675",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 2463.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-676",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2463.0, 147.0, 22.0 ],
					"text" : "/vis/edgespecularscale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-677",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2465.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1735.5, 20.0, 20.0 ],
					"varname" : "edgeSpecularScale_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-678",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2465.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1735.5, 178.0, 20.0 ],
					"text" : "Edge Specular Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-679",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2468.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1736.5, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-680",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2438.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-681",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2438.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-682",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2438.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-683",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2435.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1714.0, 50.0, 22.0 ],
					"varname" : "edgeDiffuseScale_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-684",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 2435.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1714.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeDiffuseScale_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-685",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2435.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1714.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeDiffuseScale"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-686",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 2433.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-687",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2433.0, 136.0, 22.0 ],
					"text" : "/vis/edgediffusescale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-688",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2435.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1714.0, 20.0, 20.0 ],
					"varname" : "edgeDiffuseScale_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-689",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2435.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1714.0, 178.0, 20.0 ],
					"text" : "Edge Diffusion Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-690",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2438.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1715.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-691",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2408.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-692",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2408.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-693",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2408.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-694",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2405.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1689.0, 50.0, 22.0 ],
					"varname" : "edgeAmbientScale_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-695",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 2405.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1689.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeAmbientScale_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-696",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2405.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1689.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeAmbientScale"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-697",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 2403.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-698",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2403.0, 144.0, 22.0 ],
					"text" : "/vis/edgeambientscale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-699",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2405.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1689.0, 20.0, 20.0 ],
					"varname" : "edgeAmbientScale_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-700",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2405.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1689.0, 178.0, 20.0 ],
					"text" : "Edge Ambient Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-701",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2408.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1690.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-702",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2528.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-703",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2528.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-704",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2528.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-705",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2525.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1785.0, 50.0, 22.0 ],
					"varname" : "edgeOcclusionScale_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-706",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 2525.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1785.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeOcclusionScale_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-707",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2525.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1785.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeOcclusionScale"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-708",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 2523.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-709",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2523.0, 151.0, 22.0 ],
					"text" : "/vis/edgeocclusionscale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-710",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2525.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1785.0, 20.0, 20.0 ],
					"varname" : "edgeOcclusionScale_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-711",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2525.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1785.0, 178.0, 20.0 ],
					"text" : "Edge Occlusion Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-712",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2528.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1786.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-713",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2588.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-714",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2588.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-715",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2588.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-716",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2585.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1831.0, 50.0, 22.0 ],
					"varname" : "edgeOcclusionResol_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-717",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 2585.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1831.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeOcclusionResol_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-718",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2585.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1831.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeOcclusionResol"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-719",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 2583.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-720",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2583.0, 175.0, 22.0 ],
					"text" : "/vis/edgeocclusionresolution $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-721",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2585.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1831.0, 20.0, 20.0 ],
					"varname" : "edgeOcclusionResol_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-722",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2585.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1831.0, 178.0, 20.0 ],
					"text" : "Edge Occlusion Resolution"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-723",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2588.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1832.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-724",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2558.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-725",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2558.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-726",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2558.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-727",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2555.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1808.0, 50.0, 22.0 ],
					"varname" : "edgeOcclusionRange_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-728",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 2555.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1808.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeOcclusionRange_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-729",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2555.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1808.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeOcclusionRange"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-730",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 2553.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-731",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2553.0, 154.0, 22.0 ],
					"text" : "/vis/edgeocclusionrange $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-732",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2555.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1808.0, 20.0, 20.0 ],
					"varname" : "edgeOcclusionRange_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-733",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2555.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1808.0, 178.0, 20.0 ],
					"text" : "Edge Occlusion Range"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-734",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2558.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1809.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-224",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 2377.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-225",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2369.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1661.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeOcclusionColorB"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-226",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2377.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1662.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-577",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 2345.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-634",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2337.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1638.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeOcclusionColorG"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-635",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2345.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1639.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-636",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 2315.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-637",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2307.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1615.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeOcclusionColorR"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-638",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1117.0, 2320.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-639",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 920.0, 2320.0, 183.0, 22.0 ],
					"text" : "/vis/edgeocclusioncolor $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-640",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 818.0, 2320.0, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-641",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2307.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 1615.81817626953125, 20.0, 20.0 ],
					"varname" : "edgeOcclusionColor_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-642",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2307.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1615.81817626953125, 178.0, 20.0 ],
					"text" : "Edge Occlusion Color"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-643",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2315.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1616.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-644",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 2280.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-645",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2272.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1591.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeColorB"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-646",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2280.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1592.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-647",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 2248.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-648",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2240.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1568.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeColorG"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-649",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2248.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1569.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-650",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 2218.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-651",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2210.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1545.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeColorR"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-652",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1112.0, 2223.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-653",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 920.0, 2223.0, 133.0, 22.0 ],
					"text" : "/vis/edgecolor $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-654",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 818.0, 2223.0, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-655",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2210.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 1545.81817626953125, 20.0, 20.0 ],
					"varname" : "edgeColor_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-656",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2210.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1545.81817626953125, 178.0, 20.0 ],
					"text" : "Edge Color"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-657",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2218.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1546.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-620",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 1758.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-621",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1750.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1313.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointOcclusionColorB"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-622",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1758.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1314.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-623",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 1726.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-624",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1718.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1290.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointOcclusionColorG"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-625",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1726.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1291.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-626",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 1696.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-627",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1688.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1267.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointOcclusionColorR"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-628",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1117.0, 1701.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-629",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 920.0, 1701.0, 178.0, 22.0 ],
					"text" : "/vis/jointocclusioncolor $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-630",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 818.0, 1701.0, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-631",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1688.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 1267.81817626953125, 20.0, 20.0 ],
					"varname" : "jointOcclusionColor_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-632",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1688.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1267.81817626953125, 178.0, 20.0 ],
					"text" : "Joint Occlusion Color"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-633",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1696.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1268.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-606",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 682.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-607",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 674.0, 46.0, 22.0 ],
					"triangle" : 0,
					"varname" : "BgOcclusionColorB"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-608",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 682.5, 320.0, 17.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-609",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 650.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-610",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 642.0, 46.0, 22.0 ],
					"triangle" : 0,
					"varname" : "BgOcclusionColorG"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-611",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 650.5, 320.0, 17.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-612",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 620.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-613",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 612.0, 46.0, 22.0 ],
					"triangle" : 0,
					"varname" : "BgOcclusionColorR"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-614",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1143.25, 630.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-615",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 920.25, 625.0, 169.0, 22.0 ],
					"text" : "/vis/bgocclusioncolor $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-616",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 818.25, 625.0, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-617",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 612.0, 20.0, 20.0 ],
					"varname" : "BgOcclusionColor_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-618",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 612.0, 178.0, 20.0 ],
					"text" : "Background Occlusion Color"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-619",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 620.5, 320.0, 17.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-470",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.25, 307.2236328125, 100.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-471",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 307.2236328125, 112.0, 22.0 ],
					"text" : "/mocap/rootzero $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-469",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 260.25, 305.2236328125, 24.0, 24.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 181.0, 24.0, 24.0 ],
					"varname" : "root_zero"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-467",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 305.2236328125, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 181.0, 178.0, 20.0 ],
					"text" : "Rooz Zero"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-303",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1446.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-457",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1446.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-458",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1446.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-459",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1443.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 473.81817626953125, 50.0, 22.0 ],
					"varname" : "camAngle_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-460",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1443.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 473.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "camAngle_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-461",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1443.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 473.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "camAngle"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-462",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.0, 1441.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-463",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 1441.0, 99.0, 22.0 ],
					"text" : "/vis/camangle $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-464",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1443.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 473.81817626953125, 20.0, 20.0 ],
					"varname" : "camAngle_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-465",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1443.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 473.81817626953125, 178.0, 20.0 ],
					"text" : "Camera Angle"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-466",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1446.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 474.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-59",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 1661.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-60",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1653.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1244.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointColorB"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-62",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1661.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1245.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-63",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 1629.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-66",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1621.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1221.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointColorG"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-69",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1629.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1222.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-71",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 1599.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-74",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1591.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1198.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointColorR"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-75",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1112.0, 1604.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-76",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 920.0, 1604.0, 128.0, 22.0 ],
					"text" : "/vis/jointcolor $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-77",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 818.0, 1604.0, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-78",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1591.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 1198.81817626953125, 20.0, 20.0 ],
					"varname" : "jointColor_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-79",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1591.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1198.81817626953125, 178.0, 20.0 ],
					"text" : "Joint Color"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-80",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1599.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1199.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-272",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1879.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-273",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1879.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-274",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1879.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-275",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1876.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1411.5, 50.0, 22.0 ],
					"varname" : "jointSpecularPow_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-276",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1876.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1411.5, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointSpecularPow_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-277",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1876.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1411.5, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointSpecularPow"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-279",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 1874.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-281",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 1874.0, 136.0, 22.0 ],
					"text" : "/vis/jointspecularpow $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-282",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1876.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1411.5, 20.0, 20.0 ],
					"varname" : "jointSpecularPow_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-283",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1876.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1411.5, 178.0, 20.0 ],
					"text" : "Joint Specular Pow"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-284",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1879.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1412.5, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-238",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1849.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-239",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1849.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-240",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1849.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-241",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1846.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1386.5, 50.0, 22.0 ],
					"varname" : "jointSpecularScale_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-242",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1846.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1386.5, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointSpecularScale_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-252",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1846.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1386.5, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointSpecularScale"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-256",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 1844.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-257",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 1844.0, 142.0, 22.0 ],
					"text" : "/vis/jointspecularscale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-258",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1846.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1386.5, 20.0, 20.0 ],
					"varname" : "jointSpecularScale_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-259",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1846.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1386.5, 178.0, 20.0 ],
					"text" : "Joint Specular Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-271",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1849.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1387.5, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-235",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1183.0, 1537.0, 107.0, 22.0 ],
					"text" : "expr $f3 * cos($f2)"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-236",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1183.0, 1506.0, 155.0, 22.0 ],
					"text" : "expr $f3 * sin($f1) * sin($f2)"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-237",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1183.0, 1482.0, 158.0, 22.0 ],
					"text" : "expr $f3 * cos($f1) * sin($f2)"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-232",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1183.0, 1373.0, 107.0, 22.0 ],
					"text" : "expr $f3 * cos($f2)"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-233",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1183.0, 1345.0, 155.0, 22.0 ],
					"text" : "expr $f3 * sin($f1) * sin($f2)"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-234",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1182.0, 1393.0, 158.0, 22.0 ],
					"text" : "expr $f3 * cos($f1) * sin($f2)"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-223",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1366.0, 1346.0, 94.0, 22.0 ],
					"text" : "pack 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-222",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1366.0, 1484.0, 94.0, 22.0 ],
					"text" : "pack 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-102",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2828.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-103",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2828.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-104",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2828.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-105",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2825.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 1166.403620958328247, 50.0, 22.0 ],
					"varname" : "jointEdgeSmooth_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-106",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2825.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 1166.403620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointEdgeSmooth"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-107",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.0, 2823.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-108",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2823.0, 135.0, 22.0 ],
					"text" : "/vis/jointedgesmooth $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-109",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2825.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 1166.403620958328247, 20.0, 20.0 ],
					"varname" : "jointEdgeSmooth_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-110",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2825.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1166.403620958328247, 178.0, 20.0 ],
					"text" : "Joint Edge Smooth"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-112",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2828.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 1167.403620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-146",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1548.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-147",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1548.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-148",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1548.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-149",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1545.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 349.0, 543.81817626953125, 50.0, 22.0 ],
					"varname" : "lightPosZ_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-150",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1545.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 298.0, 543.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "lightPosZ_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-151",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1545.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 246.0, 543.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "lightPosZ"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-152",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1548.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 401.0, 544.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-153",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1518.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-154",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1518.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-155",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1518.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-156",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1515.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 349.0, 520.81817626953125, 50.0, 22.0 ],
					"varname" : "lightPosY_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-157",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1515.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 298.0, 520.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "lightPosY_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-158",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1515.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 246.0, 520.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "lightPosY"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-159",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1518.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 401.0, 521.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-160",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1488.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-161",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1488.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-178",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1488.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-181",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1485.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 349.0, 497.81817626953125, 50.0, 22.0 ],
					"varname" : "lightPosX_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-182",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1485.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 298.0, 497.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "lightPosX_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-198",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1485.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 246.0, 497.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "lightPosX"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-199",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1641.0, 1484.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-200",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1467.0, 1484.0, 144.0, 22.0 ],
					"text" : "/vis/lightposition $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-201",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1081.0, 1488.0, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-202",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1485.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 497.81817626953125, 20.0, 20.0 ],
					"varname" : "lightPos_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-208",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1485.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 497.81817626953125, 178.0, 20.0 ],
					"text" : "Light Position"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-209",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1488.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 401.0, 498.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-135",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1819.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-136",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1819.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-137",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1819.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-138",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1816.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1365.0, 50.0, 22.0 ],
					"varname" : "diffuseScale_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-139",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1816.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1365.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointDiffuseScale_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-140",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1816.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1365.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointDiffuseScale"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-141",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 1814.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-142",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 1814.0, 132.0, 22.0 ],
					"text" : "/vis/jointdiffusescale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-143",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1816.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1365.0, 20.0, 20.0 ],
					"varname" : "jointDiffuseScale_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-144",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1816.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1365.0, 178.0, 20.0 ],
					"text" : "Joint Diffusion Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-145",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1819.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1366.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-124",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1789.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-125",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1789.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-126",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1789.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-127",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1786.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1340.0, 50.0, 22.0 ],
					"varname" : "jointAmbientScale_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-128",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1786.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1340.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointAmbientScale_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-129",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1786.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1340.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointAmbientScale"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-130",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 1784.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-131",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 1784.0, 139.0, 22.0 ],
					"text" : "/vis/jointambientscale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-132",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1786.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1340.0, 20.0, 20.0 ],
					"varname" : "jointAmbientScale_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-133",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1786.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1340.0, 178.0, 20.0 ],
					"text" : "Joint Ambient Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-134",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1789.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1341.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-53",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1909.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-55",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1909.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-56",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1909.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-57",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1906.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1436.0, 50.0, 22.0 ],
					"varname" : "jointOcclusionScale_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-117",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1906.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1436.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointOcclusionScale_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-118",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1906.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1436.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointOcclusionScale"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-119",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 1904.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-120",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 1904.0, 147.0, 22.0 ],
					"text" : "/vis/jointocclusionscale $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-121",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1906.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1436.0, 20.0, 20.0 ],
					"varname" : "jointOcclusionScale_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-122",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1906.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1436.0, 178.0, 20.0 ],
					"text" : "Joint Occlusion Scale"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-123",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1909.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1437.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-16",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 773.5, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-17",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 773.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-18",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 773.5, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-19",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 771.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 826.0, 50.0, 22.0 ],
					"varname" : "skeletonPosZ_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-20",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 771.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 826.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "skeletonPosZ_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-21",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 771.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 826.0, 50.0, 22.0 ],
					"triangle" : 0
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-22",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 773.5, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 827.0, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "skeletonPosZ"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-23",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 743.5, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-24",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 743.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-25",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 743.5, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-26",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 741.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 803.0, 50.0, 22.0 ],
					"varname" : "skeletonPosY_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-27",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 741.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 803.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "skeletonPosY_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-28",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 741.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 803.0, 50.0, 22.0 ],
					"triangle" : 0
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-32",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 743.5, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 804.0, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "skeletonPosY"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-33",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 713.5, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-34",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 713.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-35",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 713.5, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-36",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 711.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 780.0, 50.0, 22.0 ],
					"varname" : "skeletonPosX_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-37",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.25, 711.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 780.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "skeletonPosX_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-38",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 711.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 780.0, 50.0, 22.0 ],
					"triangle" : 0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-39",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.25, 708.5, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-41",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 708.5, 171.0, 22.0 ],
					"text" : "/mocap/skelposworld $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-45",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1077.25, 713.5, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-46",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 711.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 780.0, 20.0, 20.0 ],
					"varname" : "skeletonPos_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-47",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 711.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 780.0, 178.0, 20.0 ],
					"text" : "Skeleton Position"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-52",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 713.5, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 781.0, 320.0, 20.0 ],
					"size" : 1.0,
					"varname" : "skeletonPosX"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-15",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 129.0, 30.0, 93.0, 22.0 ],
					"text" : "r vis_preset_set"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-3",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.25, 270.7236328125, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.25, 270.7236328125, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-5",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.25, 270.7236328125, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-7",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.25, 268.2236328125, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 151.0, 50.0, 22.0 ],
					"varname" : "posUpdateSmoothing_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-8",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 268.2236328125, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 151.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "posUpdateSmoothing"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-9",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.25, 265.7236328125, 100.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-10",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.25, 265.7236328125, 160.0, 22.0 ],
					"text" : "/mocap/updatesmoothing $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-11",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 268.2236328125, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 151.0, 20.0, 20.0 ],
					"varname" : "posUpdateSmoothing_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-12",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 268.2236328125, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 144.0, 178.0, 20.0 ],
					"text" : "Update Smoothing"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-13",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 270.7236328125, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 152.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 31.0, 357.0, 153.0, 34.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 17.0, 21.18182373046875, 243.0, 20.0 ],
					"text" : "Raymarching Visualisation Control"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-407",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2768.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-408",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2768.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-409",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2768.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-410",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2765.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 1138.403620958328247, 50.0, 22.0 ],
					"varname" : "edgeSmooth_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-411",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2765.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 1138.403620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeSmooth"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-412",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.0, 2763.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-413",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2763.0, 113.0, 22.0 ],
					"text" : "/vis/edgesmooth $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-414",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2765.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 1138.403620958328247, 20.0, 20.0 ],
					"varname" : "edgeSmooth_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-415",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2765.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1138.403620958328247, 178.0, 20.0 ],
					"text" : "Edge Smooth"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-416",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2768.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 1139.403620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-417",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2738.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-418",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2738.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-419",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2738.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-420",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2735.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 1115.903620958328247, 50.0, 22.0 ],
					"varname" : "edgeRadius_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-421",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2735.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 1115.903620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeRadius"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-422",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.0, 2733.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-423",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2733.0, 104.0, 22.0 ],
					"text" : "/vis/edgeround $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-424",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2735.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 1115.903620958328247, 20.0, 20.0 ],
					"varname" : "edgeRadius_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-425",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2735.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1115.903620958328247, 178.0, 20.0 ],
					"text" : "Edge Round"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-426",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2738.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 1116.903620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-427",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2707.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-428",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2707.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-429",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2707.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-430",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2704.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 1092.903620958328247, 50.0, 22.0 ],
					"varname" : "edgeSizeZ_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-431",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2704.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 1092.903620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeSizeZ"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-432",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2707.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 1093.903620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-433",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2677.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-434",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2677.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-435",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2677.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-436",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2674.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 1069.903620958328247, 50.0, 22.0 ],
					"varname" : "edgeSizeY_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-437",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2674.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 1069.903620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeSizeY"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-438",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2677.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 1070.903620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-439",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1076.0, 2647.0, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-440",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2647.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-441",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2647.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-442",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2647.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-443",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2644.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 1046.903620958328247, 50.0, 22.0 ],
					"varname" : "edgeSizeX_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-444",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2644.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 1046.903620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "edgeSizeX"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-445",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.0, 2642.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-446",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2642.0, 128.0, 22.0 ],
					"text" : "/vis/edgesize $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-447",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2644.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 1046.903620958328247, 20.0, 20.0 ],
					"varname" : "edgeSize_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-448",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2644.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1046.903620958328247, 178.0, 20.0 ],
					"text" : "Edge Size"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-449",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2647.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 1047.903620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-451",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2618.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-452",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.0, 2613.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-453",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2613.0, 129.0, 22.0 ],
					"text" : "/vis/edgeprimitive $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-454",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2615.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 1022.903620958328247, 20.0, 20.0 ],
					"varname" : "edgePrimitive_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-455",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2615.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 1022.903620958328247, 178.0, 20.0 ],
					"text" : "Edge Primitive"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-456",
					"maxclass" : "slider",
					"min" : -1.0,
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2618.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 1023.903620958328247, 320.0, 20.0 ],
					"size" : 4.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-397",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2158.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-398",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2158.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-399",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2158.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-400",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2155.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 968.403620958328247, 50.0, 22.0 ],
					"varname" : "jointSmooth_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-401",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2155.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 968.403620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointSmooth"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-402",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.0, 2153.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-403",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2153.0, 108.0, 22.0 ],
					"text" : "/vis/jointsmooth $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-404",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2155.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 968.403620958328247, 20.0, 20.0 ],
					"varname" : "jointSmooth_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-405",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2155.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 968.403620958328247, 178.0, 20.0 ],
					"text" : "Joint Smooth"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-406",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2158.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 969.403620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-347",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2128.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-348",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2128.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-349",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2128.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-350",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2125.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 945.903620958328247, 50.0, 22.0 ],
					"varname" : "jointRound_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-351",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2125.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 945.903620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointRound"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-352",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.0, 2123.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-353",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2123.0, 101.0, 22.0 ],
					"text" : "/vis/jointround $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-354",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2125.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 945.903620958328247, 20.0, 20.0 ],
					"varname" : "jointRound_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-355",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2125.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 945.903620958328247, 178.0, 20.0 ],
					"text" : "Joint Round"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-356",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2128.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 946.903620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-341",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2097.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-342",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2097.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-343",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2097.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-344",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2094.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 922.903620958328247, 50.0, 22.0 ],
					"varname" : "jointSizeZ_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-345",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2094.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 922.903620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointSizeZ"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-346",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2097.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 923.903620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-335",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2067.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-336",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2067.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-337",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2067.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-338",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2064.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 899.903620958328247, 50.0, 22.0 ],
					"varname" : "jointSizeY_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-339",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2064.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 899.903620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointSizeY"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-340",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2067.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 900.903620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-334",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1076.0, 2037.0, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-324",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 2037.0, 78.0, 19.0 ],
					"text" : "expr ($f1) / ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-325",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 2037.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-326",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 2037.0, 73.0, 19.0 ],
					"text" : "expr $f1 * ( $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-327",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 2034.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 876.903620958328247, 50.0, 22.0 ],
					"varname" : "jointSizeX_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-328",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 2034.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 244.0, 876.903620958328247, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointSizeX"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-329",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.0, 2032.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-330",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 2032.0, 123.0, 22.0 ],
					"text" : "/vis/jointsize $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-331",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 2034.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 876.903620958328247, 20.0, 20.0 ],
					"varname" : "jointSize_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-332",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 2034.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 876.903620958328247, 178.0, 20.0 ],
					"text" : "joint Size"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-333",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2037.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 877.903620958328247, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-308",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1999.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-309",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1418.0, 1994.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-310",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 1994.0, 129.0, 22.0 ],
					"text" : "/vis/jointprimitive $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-311",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1996.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 851.903620958328247, 20.0, 20.0 ],
					"varname" : "jointPrimitive_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-312",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1996.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 851.903620958328247, 178.0, 20.0 ],
					"text" : "Joint Primitive"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-313",
					"maxclass" : "slider",
					"min" : -1.0,
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 2004.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 399.0, 852.903620958328247, 320.0, 20.0 ],
					"size" : 4.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-260",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1969.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-261",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1969.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-262",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1969.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-263",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1966.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1482.0, 50.0, 22.0 ],
					"varname" : "jointOcclusionResol_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-264",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1966.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1482.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointOcclusionResol_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-265",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1966.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1482.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointOcclusionResol"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-266",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 1964.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-267",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 1964.0, 171.0, 22.0 ],
					"text" : "/vis/jointocclusionresolution $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-268",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1966.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1482.0, 20.0, 20.0 ],
					"varname" : "jointOcclusionResol_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-269",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1966.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1482.0, 178.0, 20.0 ],
					"text" : "Joint Occlusion Resolution"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-270",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1969.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1483.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-243",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1939.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-244",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1939.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-245",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1939.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-246",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1936.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 1459.0, 50.0, 22.0 ],
					"varname" : "jointOcclusionRange_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-247",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1936.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 1459.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointOcclusionRange_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-248",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1936.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 1459.0, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "jointOcclusionRange"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-249",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1616.0, 1934.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-250",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1244.0, 1934.0, 149.0, 22.0 ],
					"text" : "/vis/jointocclusionrange $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-253",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1936.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 1459.0, 20.0, 20.0 ],
					"varname" : "jointOcclusionRange_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-254",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1936.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 58.0, 1459.0, 178.0, 20.0 ],
					"text" : "Joint Occlusion Range"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-255",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1939.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 1460.0, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-190",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1406.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-191",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1406.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-192",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1406.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-193",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1403.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 450.81817626953125, 50.0, 22.0 ],
					"varname" : "camPosZ_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-195",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1403.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 450.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "camPosZ_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-196",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1403.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 450.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "camPosZ"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-197",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1406.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 451.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-183",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1376.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-184",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1376.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-185",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1376.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-186",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1373.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 427.81817626953125, 50.0, 22.0 ],
					"varname" : "camPosY_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-187",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1373.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 427.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "camPosY_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-188",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1373.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 427.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "camPosY"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-189",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1376.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 428.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-180",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 1346.0, 119.0, 19.0 ],
					"text" : "expr ($f1 - $f2) / ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-179",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1019.0, 1346.0, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-177",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 757.0, 1346.0, 118.0, 19.0 ],
					"text" : "expr $f2 + $f1 * ( $f3 - $f2 )"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-175",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 367.0, 1343.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 404.81817626953125, 50.0, 22.0 ],
					"varname" : "camPosX_max"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-176",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 315.0, 1343.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 297.0, 404.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "camPosX_min"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-162",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.0, 1343.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 404.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "camPosX"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-165",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1641.0, 1346.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-170",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1467.0, 1346.0, 144.0, 22.0 ],
					"text" : "/vis/camposition $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-171",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1081.0, 1346.0, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-172",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 1343.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 404.81817626953125, 20.0, 20.0 ],
					"varname" : "camPos_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-173",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 1343.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 404.81817626953125, 178.0, 20.0 ],
					"text" : "Camera Position"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-174",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.0, 1346.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 405.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-95",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 573.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-96",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 565.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 733.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "BgColorS"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-97",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 573.5, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 734.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-67",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 541.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-68",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 533.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 710.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "BgColorG"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-70",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 541.5, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 711.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-64",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 763.0, 511.5, 41.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-31",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 264.25, 503.0, 46.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 245.0, 687.81817626953125, 50.0, 22.0 ],
					"triangle" : 0,
					"varname" : "BgColorR"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-6",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1143.25, 521.0, 91.0, 22.0 ],
					"text" : "s vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-29",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 920.25, 516.0, 119.0, 22.0 ],
					"text" : "/vis/bgcolor $1 $2 $3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-30",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 818.25, 516.0, 88.0, 22.0 ],
					"text" : "pak 0.1 0.1 0.1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-72",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 31.0, 503.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 28.0, 687.81817626953125, 20.0, 20.0 ],
					"varname" : "BgColor_interest"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-73",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 503.0, 178.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 57.0, 687.81817626953125, 178.0, 20.0 ],
					"text" : "Background Color"
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-65",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 425.25, 511.5, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 688.81817626953125, 320.0, 20.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-251",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 652.0, 129.0, 95.0, 22.0 ],
					"text" : "expr $f1 * 1000."
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"id" : "obj-43",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 652.0, 43.0, 42.0, 19.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 568.5, 67.0, 42.0, 19.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-194",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 652.0, 74.0, 43.0, 22.0 ],
					"text" : "set $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-111",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 625.0, 43.0, 20.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 541.5, 67.0, 20.0, 20.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-44",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 652.0, 99.0, 50.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"format" : 6,
					"id" : "obj-48",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 652.0, 222.0, 50.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-93",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 652.0, 198.0, 104.0, 22.0 ],
					"text" : "expr $f1 / 100000"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-88",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 652.0, 153.0, 86.0, 23.0 ],
					"text" : "0, 100000 $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-49",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"patching_rect" : [ 652.0, 175.0, 64.0, 22.0 ],
					"text" : "line 1 100"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 11.0,
					"hidden" : 1,
					"id" : "obj-50",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "int", "bang" ],
					"patching_rect" : [ 260.25, 72.0, 32.5, 21.0 ],
					"text" : "t i b"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 11.0,
					"hidden" : 1,
					"id" : "obj-51",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 260.25, 99.0, 45.0, 21.0 ],
					"text" : "pack f i"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 11.0,
					"hidden" : 1,
					"id" : "obj-54",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 360.25, 121.0, 62.0, 21.0 ],
					"text" : "store 1000"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 11.0,
					"hidden" : 1,
					"id" : "obj-58",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 260.25, 121.0, 95.0, 21.0 ],
					"text" : "recall 1000 $2 $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"hidden" : 1,
					"id" : "obj-61",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 250.0, 18.0, 32.5, 22.0 ],
					"text" : "0"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 9.0,
					"id" : "obj-84",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 250.0, 43.0, 41.0, 19.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 166.5, 67.0, 41.0, 19.0 ]
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-85",
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"orientation" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 296.0, 43.0, 320.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 212.5, 67.0, 320.0, 17.0 ],
					"size" : 1.0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"hidden" : 1,
					"id" : "obj-280",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 55.25, 64.0, 103.0, 22.0 ],
					"text" : "read controls.json"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"hidden" : 1,
					"id" : "obj-278",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 62.0, 30.0, 60.0, 22.0 ],
					"text" : "loadbang"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Verdana Bold",
					"fontsize" : 9.0,
					"id" : "obj-163",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 50.0, 92.0, 48.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 17.0, 48.0, 48.0, 17.0 ],
					"text" : "presets"
				}

			}
, 			{
				"box" : 				{
					"angle" : 0.0,
					"bgcolor" : [ 0.941176, 0.941176, 0.941176, 1.0 ],
					"border" : 1,
					"bordercolor" : [ 0.823529, 0.823529, 0.823529, 1.0 ],
					"hidden" : 1,
					"id" : "obj-164",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 49.0, 92.0, 138.0, 18.0 ],
					"proportion" : 0.39,
					"rounded" : 0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Verdana",
					"fontsize" : 9.0,
					"id" : "obj-166",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 125.0, 115.0, 36.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 109.5, 48.0, 36.0, 17.0 ],
					"text" : "recall"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Verdana",
					"fontsize" : 9.0,
					"id" : "obj-167",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 62.0, 115.0, 34.0, 17.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 70.0, 48.0, 34.0, 17.0 ],
					"text" : "store"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Verdana",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-168",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 61.0, 161.0, 44.0, 17.0 ],
					"text" : "presets"
				}

			}
, 			{
				"box" : 				{
					"allowdrag" : 0,
					"fontname" : "Verdana",
					"fontsize" : 9.0,
					"id" : "obj-169",
					"items" : [ "read", ",", "write", ",", "clientwindow", ",", "storagewindow" ],
					"labelclick" : 1,
					"maxclass" : "umenu",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "", "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 61.0, 180.0, 78.0, 19.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 70.0, 90.0, 87.5, 19.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Verdana",
					"fontsize" : 9.0,
					"format" : 6,
					"id" : "obj-203",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 125.0, 135.0, 48.0, 19.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 109.5, 67.0, 48.0, 19.0 ],
					"triscale" : 0.9
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Verdana",
					"fontsize" : 9.0,
					"id" : "obj-204",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 77.0, 135.0, 35.0, 19.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 70.0, 67.0, 35.0, 19.0 ],
					"triscale" : 0.9
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-205",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 61.0, 135.0, 19.0, 19.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 48.0, 67.0, 19.0, 19.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Verdana",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-206",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 61.0, 156.0, 66.0, 19.0 ],
					"text" : "pack store 0"
				}

			}
, 			{
				"box" : 				{
					"autorestore" : "controls.json",
					"fontname" : "Verdana",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-207",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 61.0, 200.0, 67.0, 19.0 ],
					"saved_object_attributes" : 					{
						"client_rect" : [ 425, 64, 1733, 841 ],
						"parameter_enable" : 0,
						"parameter_mappable" : 0,
						"storage_rect" : [ 8, 65, 1687, 1125 ]
					}
,
					"text" : "pattrstorage",
					"varname" : "u845000342"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Verdana",
					"fontsize" : 9.0,
					"hidden" : 1,
					"id" : "obj-211",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "", "", "", "" ],
					"patching_rect" : [ 129.0, 200.0, 53.0, 19.0 ],
					"restore" : 					{
						"BgColorG" : [ 0.0 ],
						"BgColorR" : [ 0.0 ],
						"BgColorS" : [ 0.0 ],
						"BgColor_interest" : [ 0 ],
						"BgOcclusionColorB" : [ 0.454838709677419 ],
						"BgOcclusionColorG" : [ 0.506451612903226 ],
						"BgOcclusionColorR" : [ 0.199835526315789 ],
						"BgOcclusionColor_interest" : [ 0 ],
						"bg_active" : [ 0 ],
						"camAngle" : [ 36.711599999999997 ],
						"camAngle_interest" : [ 0 ],
						"camAngle_max" : [ 120.0 ],
						"camAngle_min" : [ 5.0 ],
						"camPosX" : [ -1.906331027164686 ],
						"camPosX_max" : [ 3.141 ],
						"camPosX_min" : [ -3.141 ],
						"camPosY" : [ -1.689687351443124 ],
						"camPosY_max" : [ 3.141 ],
						"camPosY_min" : [ -3.141 ],
						"camPosZ" : [ -3.368208828522921 ],
						"camPosZ_max" : [ 10.0 ],
						"camPosZ_min" : [ -20.0 ],
						"camPos_interest" : [ 0 ],
						"diffuseScale_max" : [ 1.0 ],
						"edgeAmbientScale" : [ 0.01 ],
						"edgeAmbientScale_interest" : [ 0 ],
						"edgeAmbientScale_max" : [ 1.0 ],
						"edgeAmbientScale_min" : [ 0.0 ],
						"edgeColorB" : [ 0.6873 ],
						"edgeColorG" : [ 0.5571 ],
						"edgeColorR" : [ 0.6355 ],
						"edgeColor_interest" : [ 0 ],
						"edgeDiffuseScale" : [ 0.995 ],
						"edgeDiffuseScale_interest" : [ 0 ],
						"edgeDiffuseScale_max" : [ 1.0 ],
						"edgeDiffuseScale_min" : [ 0.0 ],
						"edgeOcclusionColorB" : [ 0.0 ],
						"edgeOcclusionColorG" : [ 0.0 ],
						"edgeOcclusionColorR" : [ 0.0 ],
						"edgeOcclusionColor_interest" : [ 0 ],
						"edgeOcclusionRange" : [ 1.11 ],
						"edgeOcclusionRange_interest" : [ 0 ],
						"edgeOcclusionRange_max" : [ 10.0 ],
						"edgeOcclusionRange_min" : [ 0.0 ],
						"edgeOcclusionResol" : [ 0.38 ],
						"edgeOcclusionResol_interest" : [ 0 ],
						"edgeOcclusionResol_max" : [ 1.0 ],
						"edgeOcclusionResol_min" : [ 0.0 ],
						"edgeOcclusionScale" : [ 0.85 ],
						"edgeOcclusionScale_interest" : [ 0 ],
						"edgeOcclusionScale_max" : [ 1.0 ],
						"edgeOcclusionScale_min" : [ 0.0 ],
						"edgePrimitive" : [ 1.0 ],
						"edgePrimitive_interest" : [ 0 ],
						"edgeRadius" : [ 0.0 ],
						"edgeRadius_interest" : [ 0 ],
						"edgeRadius_max" : [ 1.0 ],
						"edgeSizeX" : [ 0.02 ],
						"edgeSizeX_max" : [ 1.0 ],
						"edgeSizeY" : [ 0.02 ],
						"edgeSizeY_max" : [ 1.0 ],
						"edgeSizeZ" : [ 1.0 ],
						"edgeSizeZ_max" : [ 4.0 ],
						"edgeSize_interest" : [ 0 ],
						"edgeSmooth" : [ 0.0 ],
						"edgeSmooth_interest" : [ 0 ],
						"edgeSmooth_max" : [ 1.0 ],
						"edgeSpecularPow" : [ 5.16 ],
						"edgeSpecularPow_interest" : [ 0 ],
						"edgeSpecularPow_max" : [ 100.0 ],
						"edgeSpecularPow_min" : [ 0.0 ],
						"edgeSpecularScale" : [ 1.877 ],
						"edgeSpecularScale_interest" : [ 0 ],
						"edgeSpecularScale_max" : [ 2.0 ],
						"edgeSpecularScale_min" : [ 0.0 ],
						"fogMaxDist" : [ 100.0 ],
						"fogMaxDist[1]" : [ 0.0 ],
						"fogMaxDist_interest" : [ 0 ],
						"fogMaxDist_interest[1]" : [ 0 ],
						"fogMaxDist_max" : [ 100.0 ],
						"fogMaxDist_max[1]" : [ 2.0 ],
						"fogMaxDist_min" : [ 0.0 ],
						"fogMaxDist_min[1]" : [ 0.0 ],
						"fogMinDist" : [ 100.0 ],
						"fogMinDist[1]" : [ 0.0 ],
						"fogMinDist_interest" : [ 0 ],
						"fogMinDist_interest[1]" : [ 0 ],
						"fogMinDist_max" : [ 100.0 ],
						"fogMinDist_max[1]" : [ 1.0 ],
						"fogMinDist_min" : [ 0.0 ],
						"fogMinDist_min[1]" : [ -1.0 ],
						"groundPosY[10]" : [ 0.5 ],
						"groundPosZ[21]" : [ 0.167699490662139 ],
						"groundPosZ[22]" : [ 0.037685215081039 ],
						"groundPrimitive[11]" : [ 1.0 ],
						"groundPrimitive_interest[11]" : [ 0 ],
						"groundRotX[10]" : [ 0.5 ],
						"groundRotY[10]" : [ 0.5 ],
						"groundRotZ[10]" : [ 0.5 ],
						"groundRound[10]" : [ 0.0 ],
						"groundRound_interest[10]" : [ 0 ],
						"groundRound_max[10]" : [ 1.0 ],
						"groundSizeX[10]" : [ 0.084485262323721 ],
						"groundSizeX_max[10]" : [ 1.0 ],
						"groundSizeY[10]" : [ 0.0 ],
						"groundSizeY_max[10]" : [ 1.0 ],
						"groundSizeZ[10]" : [ 0.0 ],
						"groundSizeZ_max[10]" : [ 1.0 ],
						"groundSize_interest[10]" : [ 0 ],
						"groundSmooth[10]" : [ 0.0 ],
						"groundSmooth_interest[10]" : [ 0 ],
						"groundSmooth_max[10]" : [ 1.0 ],
						"jointAmbientScale" : [ 0.01 ],
						"jointAmbientScale_interest" : [ 0 ],
						"jointAmbientScale_max" : [ 1.0 ],
						"jointAmbientScale_min" : [ 0.0 ],
						"jointColorB" : [ 0.6873 ],
						"jointColorG" : [ 0.5571 ],
						"jointColorR" : [ 0.6355 ],
						"jointColor_interest" : [ 0 ],
						"jointDiffuseScale" : [ 0.995 ],
						"jointDiffuseScale_interest" : [ 0 ],
						"jointDiffuseScale_min" : [ 0.0 ],
						"jointEdgeSmooth" : [ 0.0 ],
						"jointEdgeSmooth[1]" : [ 0.771456904595493 ],
						"jointEdgeSmooth_interest" : [ 0 ],
						"jointEdgeSmooth_interest[1]" : [ 0 ],
						"jointEdgeSmooth_max" : [ 1.0 ],
						"jointEdgeSmooth_max[1]" : [ 5.0 ],
						"jointOcclusionColorB" : [ 0.0 ],
						"jointOcclusionColorG" : [ 0.0 ],
						"jointOcclusionColorR" : [ 0.0 ],
						"jointOcclusionColor_interest" : [ 0 ],
						"jointOcclusionRange" : [ 1.11 ],
						"jointOcclusionRange_interest" : [ 0 ],
						"jointOcclusionRange_max" : [ 10.0 ],
						"jointOcclusionRange_min" : [ 0.0 ],
						"jointOcclusionResol" : [ 0.38 ],
						"jointOcclusionResol_interest" : [ 0 ],
						"jointOcclusionResol_max" : [ 1.0 ],
						"jointOcclusionResol_min" : [ 0.01 ],
						"jointOcclusionScale" : [ 0.85 ],
						"jointOcclusionScale_interest" : [ 0 ],
						"jointOcclusionScale_max" : [ 1.0 ],
						"jointOcclusionScale_min" : [ 0.0 ],
						"jointPrimitive" : [ 1.0 ],
						"jointPrimitive_interest" : [ 0 ],
						"jointRound" : [ 0.0 ],
						"jointRound_interest" : [ 0 ],
						"jointRound_max" : [ 1.0 ],
						"jointSizeX" : [ 0.05 ],
						"jointSizeX_max" : [ 4.0 ],
						"jointSizeY" : [ 0.05 ],
						"jointSizeY_max" : [ 4.0 ],
						"jointSizeZ" : [ 0.05 ],
						"jointSizeZ_max" : [ 4.0 ],
						"jointSize_interest" : [ 0 ],
						"jointSmooth" : [ 0.0 ],
						"jointSmooth_interest" : [ 0 ],
						"jointSmooth_max" : [ 1.0 ],
						"jointSpecularPow" : [ 5.16 ],
						"jointSpecularPow_interest" : [ 0 ],
						"jointSpecularPow_max" : [ 100.0 ],
						"jointSpecularPow_min" : [ 0.0 ],
						"jointSpecularScale" : [ 1.877 ],
						"jointSpecularScale_interest" : [ 0 ],
						"jointSpecularScale_max" : [ 2.0 ],
						"jointSpecularScale_min" : [ 0.0 ],
						"lightPosX" : [ -1.960325297113752 ],
						"lightPosX_max" : [ 3.141 ],
						"lightPosX_min" : [ -3.141 ],
						"lightPosY" : [ -0.974429923599321 ],
						"lightPosY_max" : [ 3.141 ],
						"lightPosY_min" : [ -3.141 ],
						"lightPosZ" : [ -50.0 ],
						"lightPosZ_max" : [ 100.0 ],
						"lightPosZ_min" : [ -100.0 ],
						"lightPos_interest" : [ 0 ],
						"objectAmbientScale[10]" : [ 0.0 ],
						"objectAmbientScale_interest[10]" : [ 0 ],
						"objectAmbientScale_max[10]" : [ 0.0 ],
						"objectAmbientScale_min[10]" : [ 0.0 ],
						"objectAmplitudeX[10]" : [ 0.041935483870968 ],
						"objectAmplitudeX_max[10]" : [ 1.0 ],
						"objectAmplitudeX_min[10]" : [ 0.0 ],
						"objectAmplitudeY[10]" : [ 0.034185775016484 ],
						"objectAmplitudeY_max[10]" : [ 1.0 ],
						"objectAmplitudeY_min[10]" : [ 0.0 ],
						"objectAmplitudeZ[10]" : [ 0.037685215081039 ],
						"objectAmplitudeZ_max[10]" : [ 1.0 ],
						"objectAmplitudeZ_min[10]" : [ 0.0 ],
						"objectAmplitude_interest[10]" : [ 0 ],
						"objectColorB[10]" : [ 1.0 ],
						"objectColorG[10]" : [ 1.0 ],
						"objectColorR[10]" : [ 1.0 ],
						"objectColor_interest[10]" : [ 0 ],
						"objectDiffuseScale[10]" : [ 0.0 ],
						"objectDiffuseScale_interest[10]" : [ 0 ],
						"objectDiffuseScale_max[10]" : [ 0.0 ],
						"objectDiffuseScale_min[10]" : [ 0.0 ],
						"objectFrequencyX[10]" : [ 3.141 ],
						"objectFrequencyX_max[10]" : [ 3.141 ],
						"objectFrequencyX_min[10]" : [ 0.0 ],
						"objectFrequencyY[10]" : [ 3.141 ],
						"objectFrequencyY_max[10]" : [ 3.141 ],
						"objectFrequencyY_min[10]" : [ 0.0 ],
						"objectFrequencyZ[10]" : [ 3.141 ],
						"objectFrequencyZ_max[10]" : [ 3.141 ],
						"objectFrequencyZ_min[10]" : [ 0.0 ],
						"objectFrequency_interest[10]" : [ 0 ],
						"objectOcclusionColorB[10]" : [ 0.0 ],
						"objectOcclusionColorG[10]" : [ 0.0 ],
						"objectOcclusionColorR[10]" : [ 0.0 ],
						"objectOcclusionColor_interest[10]" : [ 0 ],
						"objectOcclusionRange[10]" : [ 0.0 ],
						"objectOcclusionRange_interest[10]" : [ 0 ],
						"objectOcclusionRange_max[10]" : [ 0.0 ],
						"objectOcclusionRange_min[10]" : [ 0.0 ],
						"objectOcclusionResol[10]" : [ 0.0 ],
						"objectOcclusionResol_interest[10]" : [ 0 ],
						"objectOcclusionResol_max[10]" : [ 0.0 ],
						"objectOcclusionResol_min[10]" : [ 0.0 ],
						"objectOcclusionScale[10]" : [ 0.0 ],
						"objectOcclusionScale_interest[10]" : [ 0 ],
						"objectOcclusionScale_max[10]" : [ 0.0 ],
						"objectOcclusionScale_min[10]" : [ 0.0 ],
						"objectPhaseX[10]" : [ 2.751516 ],
						"objectPhaseX_max[10]" : [ 3.141 ],
						"objectPhaseX_min[10]" : [ -3.141 ],
						"objectPhaseY[10]" : [ 2.751516 ],
						"objectPhaseY_max[10]" : [ 3.141 ],
						"objectPhaseY_min[10]" : [ -3.141 ],
						"objectPhaseZ[10]" : [ 2.751516 ],
						"objectPhaseZ_max[10]" : [ 3.141 ],
						"objectPhaseZ_min[10]" : [ -3.141 ],
						"objectPhase_interest[10]" : [ 0 ],
						"objectSpecularPow[10]" : [ 0.0 ],
						"objectSpecularPow_interest[10]" : [ 0 ],
						"objectSpecularPow_max[10]" : [ 0.0 ],
						"objectSpecularPow_min[10]" : [ 0.0 ],
						"objectSpecularScale[10]" : [ 0.0 ],
						"objectSpecularScale_interest[10]" : [ 0 ],
						"objectSpecularScale_max[10]" : [ 0.0 ],
						"objectSpecularScale_min[10]" : [ 0.0 ],
						"object_phaseLoop[11]" : [ 1 ],
						"object_phaseLoop_duration[11]" : [ 10 ],
						"posUpdateSmoothing" : [ 0.98 ],
						"posUpdateSmoothing_interest" : [ 0 ],
						"posUpdateSmoothing_max" : [ 1.0 ],
						"root_zero" : [ 1 ],
						"scenePos_interest" : [ 0 ],
						"scenePositionX" : [ 0.0 ],
						"scenePositionX_max" : [ 20.0 ],
						"scenePositionX_min" : [ -20.0 ],
						"scenePositionY" : [ 0.0 ],
						"scenePositionY_max" : [ 20.0 ],
						"scenePositionY_min" : [ -20.0 ],
						"scenePositionZ" : [ 0.123938879456709 ],
						"scenePositionZ_max" : [ 100.0 ],
						"scenePositionZ_min" : [ -20.0 ],
						"sceneRotX" : [ 0.0 ],
						"sceneRotX_max" : [ 180.0 ],
						"sceneRotX_min" : [ -180.0 ],
						"sceneRotY" : [ 0.0 ],
						"sceneRotY_max" : [ 180.0 ],
						"sceneRotZ" : [ 0.0 ],
						"sceneRotZ_max" : [ 180.0 ],
						"sceneRotZ_min" : [ -180.0 ],
						"sceneRot_interest" : [ 0 ],
						"sceneRot_min" : [ -180.0 ],
						"shadowSmooth" : [ 0.0 ],
						"shadowSmooth_interest" : [ 0 ],
						"shadowSmooth_max" : [ 1.0 ],
						"shadowSmooth_min" : [ 0.0 ],
						"shadowStrength" : [ 0.0 ],
						"shadowStrength_interest" : [ 0 ],
						"shadowStrength_max" : [ 1.0 ],
						"shadowStrength_min" : [ 0.0 ],
						"skeletonPosX" : [ 0.5 ],
						"skeletonPosX_max" : [ 20.0 ],
						"skeletonPosX_min" : [ -20.0 ],
						"skeletonPosY" : [ 0.5 ],
						"skeletonPosY_max" : [ 20.0 ],
						"skeletonPosY_min" : [ -20.0 ],
						"skeletonPosZ" : [ 0.5 ],
						"skeletonPosZ_max" : [ 20.0 ],
						"skeletonPosZ_min" : [ -20.0 ],
						"skeletonPos_interest" : [ 0 ]
					}
,
					"text" : "autopattr",
					"varname" : "u729000346"
				}

			}
, 			{
				"box" : 				{
					"angle" : 0.0,
					"bgcolor" : [ 0.968627, 0.968627, 0.968627, 1.0 ],
					"border" : 1,
					"bordercolor" : [ 0.901961, 0.901961, 0.901961, 1.0 ],
					"hidden" : 1,
					"id" : "obj-212",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 49.0, 111.0, 138.0, 112.0 ],
					"proportion" : 0.39,
					"rounded" : 0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-40",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 800.5, 43.0, 89.0, 22.0 ],
					"text" : "r vis_osc_send"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-42",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 800.5, 81.0, 138.0, 22.0 ],
					"text" : "udpsend 127.0.0.1 9005"
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-9", 0 ],
					"source" : [ "obj-10", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-579", 0 ],
					"source" : [ "obj-1002", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-112", 0 ],
					"source" : [ "obj-102", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-106", 0 ],
					"source" : [ "obj-103", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-103", 0 ],
					"order" : 1,
					"source" : [ "obj-104", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-108", 0 ],
					"order" : 0,
					"source" : [ "obj-104", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-102", 1 ],
					"order" : 0,
					"source" : [ "obj-105", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-104", 1 ],
					"order" : 1,
					"source" : [ "obj-105", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-102", 0 ],
					"source" : [ "obj-106", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1076", 0 ],
					"source" : [ "obj-1067", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1071", 0 ],
					"source" : [ "obj-1068", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1068", 0 ],
					"order" : 1,
					"source" : [ "obj-1069", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1073", 0 ],
					"order" : 0,
					"source" : [ "obj-1069", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1067", 1 ],
					"order" : 0,
					"source" : [ "obj-1070", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1069", 1 ],
					"order" : 1,
					"source" : [ "obj-1070", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1067", 0 ],
					"source" : [ "obj-1071", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1072", 0 ],
					"source" : [ "obj-1073", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1069", 0 ],
					"source" : [ "obj-1076", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-107", 0 ],
					"source" : [ "obj-108", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-44", 0 ],
					"source" : [ "obj-111", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-104", 0 ],
					"source" : [ "obj-112", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-53", 1 ],
					"order" : 0,
					"source" : [ "obj-117", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-56", 1 ],
					"order" : 1,
					"source" : [ "obj-117", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-53", 0 ],
					"source" : [ "obj-118", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-119", 0 ],
					"source" : [ "obj-120", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-56", 0 ],
					"source" : [ "obj-123", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-134", 0 ],
					"source" : [ "obj-124", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-129", 0 ],
					"source" : [ "obj-125", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-125", 0 ],
					"order" : 1,
					"source" : [ "obj-126", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-131", 0 ],
					"order" : 0,
					"source" : [ "obj-126", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-124", 2 ],
					"order" : 0,
					"source" : [ "obj-127", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-126", 2 ],
					"order" : 1,
					"source" : [ "obj-127", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-124", 1 ],
					"order" : 0,
					"source" : [ "obj-128", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-126", 1 ],
					"order" : 1,
					"source" : [ "obj-128", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-124", 0 ],
					"source" : [ "obj-129", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 0 ],
					"source" : [ "obj-13", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-130", 0 ],
					"source" : [ "obj-131", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-126", 0 ],
					"source" : [ "obj-134", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-145", 0 ],
					"source" : [ "obj-135", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-140", 0 ],
					"source" : [ "obj-136", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-136", 0 ],
					"order" : 1,
					"source" : [ "obj-137", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-142", 0 ],
					"order" : 0,
					"source" : [ "obj-137", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-135", 2 ],
					"order" : 0,
					"source" : [ "obj-138", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-137", 2 ],
					"order" : 1,
					"source" : [ "obj-138", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-135", 1 ],
					"order" : 0,
					"source" : [ "obj-139", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-137", 1 ],
					"order" : 1,
					"source" : [ "obj-139", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-135", 0 ],
					"source" : [ "obj-140", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-141", 0 ],
					"source" : [ "obj-142", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-137", 0 ],
					"source" : [ "obj-145", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-152", 0 ],
					"source" : [ "obj-146", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-151", 0 ],
					"source" : [ "obj-147", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-147", 0 ],
					"order" : 1,
					"source" : [ "obj-148", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-201", 2 ],
					"order" : 0,
					"source" : [ "obj-148", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-146", 2 ],
					"order" : 0,
					"source" : [ "obj-149", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-148", 2 ],
					"order" : 1,
					"source" : [ "obj-149", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-203", 0 ],
					"source" : [ "obj-15", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-146", 1 ],
					"order" : 0,
					"source" : [ "obj-150", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-148", 1 ],
					"order" : 1,
					"source" : [ "obj-150", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-146", 0 ],
					"source" : [ "obj-151", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-148", 0 ],
					"source" : [ "obj-152", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-159", 0 ],
					"source" : [ "obj-153", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-158", 0 ],
					"source" : [ "obj-154", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-154", 0 ],
					"order" : 1,
					"source" : [ "obj-155", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-201", 1 ],
					"order" : 0,
					"source" : [ "obj-155", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-153", 2 ],
					"order" : 0,
					"source" : [ "obj-156", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-155", 2 ],
					"order" : 1,
					"source" : [ "obj-156", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-153", 1 ],
					"order" : 0,
					"source" : [ "obj-157", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-155", 1 ],
					"order" : 1,
					"source" : [ "obj-157", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-153", 0 ],
					"source" : [ "obj-158", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-155", 0 ],
					"source" : [ "obj-159", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-22", 0 ],
					"source" : [ "obj-16", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-209", 0 ],
					"source" : [ "obj-160", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-198", 0 ],
					"source" : [ "obj-161", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-180", 0 ],
					"source" : [ "obj-162", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-207", 0 ],
					"hidden" : 1,
					"midpoints" : [ 100.0, 151.0, 70.5, 151.0 ],
					"source" : [ "obj-169", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-21", 0 ],
					"source" : [ "obj-17", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-165", 0 ],
					"source" : [ "obj-170", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-232", 0 ],
					"order" : 0,
					"source" : [ "obj-171", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-233", 0 ],
					"order" : 1,
					"source" : [ "obj-171", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-234", 0 ],
					"order" : 2,
					"source" : [ "obj-171", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-177", 0 ],
					"source" : [ "obj-174", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-177", 2 ],
					"order" : 1,
					"source" : [ "obj-175", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-180", 2 ],
					"order" : 0,
					"source" : [ "obj-175", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-177", 1 ],
					"order" : 1,
					"source" : [ "obj-176", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-180", 1 ],
					"order" : 0,
					"source" : [ "obj-176", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-171", 0 ],
					"order" : 0,
					"source" : [ "obj-177", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-179", 0 ],
					"order" : 1,
					"source" : [ "obj-177", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-161", 0 ],
					"order" : 1,
					"source" : [ "obj-178", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-201", 0 ],
					"order" : 0,
					"source" : [ "obj-178", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-162", 0 ],
					"source" : [ "obj-179", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-17", 0 ],
					"order" : 1,
					"source" : [ "obj-18", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-45", 2 ],
					"order" : 0,
					"source" : [ "obj-18", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-174", 0 ],
					"source" : [ "obj-180", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-160", 2 ],
					"order" : 0,
					"source" : [ "obj-181", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-178", 2 ],
					"order" : 1,
					"source" : [ "obj-181", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-160", 1 ],
					"order" : 0,
					"source" : [ "obj-182", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-178", 1 ],
					"order" : 1,
					"source" : [ "obj-182", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-189", 0 ],
					"source" : [ "obj-183", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-188", 0 ],
					"source" : [ "obj-184", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-171", 1 ],
					"order" : 0,
					"source" : [ "obj-185", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-184", 0 ],
					"order" : 1,
					"source" : [ "obj-185", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-183", 2 ],
					"order" : 0,
					"source" : [ "obj-186", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-185", 2 ],
					"order" : 1,
					"source" : [ "obj-186", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-183", 1 ],
					"order" : 0,
					"source" : [ "obj-187", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-185", 1 ],
					"order" : 1,
					"source" : [ "obj-187", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-183", 0 ],
					"source" : [ "obj-188", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-185", 0 ],
					"source" : [ "obj-189", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 2 ],
					"order" : 0,
					"source" : [ "obj-19", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-18", 2 ],
					"order" : 1,
					"source" : [ "obj-19", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-197", 0 ],
					"source" : [ "obj-190", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-196", 0 ],
					"source" : [ "obj-191", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-171", 2 ],
					"order" : 0,
					"source" : [ "obj-192", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-191", 0 ],
					"order" : 1,
					"source" : [ "obj-192", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-190", 2 ],
					"order" : 0,
					"source" : [ "obj-193", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-192", 2 ],
					"order" : 1,
					"source" : [ "obj-193", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-44", 0 ],
					"source" : [ "obj-194", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-190", 1 ],
					"order" : 0,
					"source" : [ "obj-195", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-192", 1 ],
					"order" : 1,
					"source" : [ "obj-195", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-190", 0 ],
					"source" : [ "obj-196", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-192", 0 ],
					"source" : [ "obj-197", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-160", 0 ],
					"source" : [ "obj-198", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 1 ],
					"order" : 0,
					"source" : [ "obj-20", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-18", 1 ],
					"order" : 1,
					"source" : [ "obj-20", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-199", 0 ],
					"source" : [ "obj-200", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-235", 0 ],
					"order" : 0,
					"source" : [ "obj-201", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-236", 0 ],
					"order" : 1,
					"source" : [ "obj-201", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-237", 0 ],
					"order" : 2,
					"source" : [ "obj-201", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-207", 0 ],
					"hidden" : 1,
					"midpoints" : [ 134.5, 152.0, 70.5, 152.0 ],
					"source" : [ "obj-203", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-206", 1 ],
					"hidden" : 1,
					"midpoints" : [ 86.5, 151.0, 117.5, 151.0 ],
					"source" : [ "obj-204", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-206", 0 ],
					"hidden" : 1,
					"source" : [ "obj-205", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-207", 0 ],
					"hidden" : 1,
					"source" : [ "obj-206", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-178", 0 ],
					"source" : [ "obj-209", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 0 ],
					"source" : [ "obj-21", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-18", 0 ],
					"source" : [ "obj-22", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-200", 0 ],
					"source" : [ "obj-222", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-170", 0 ],
					"source" : [ "obj-223", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-225", 0 ],
					"source" : [ "obj-224", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-226", 0 ],
					"source" : [ "obj-225", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-224", 0 ],
					"order" : 1,
					"source" : [ "obj-226", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-640", 2 ],
					"order" : 0,
					"source" : [ "obj-226", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-32", 0 ],
					"source" : [ "obj-23", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-223", 2 ],
					"source" : [ "obj-232", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-223", 1 ],
					"source" : [ "obj-233", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-223", 0 ],
					"source" : [ "obj-234", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-222", 2 ],
					"source" : [ "obj-235", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-222", 1 ],
					"source" : [ "obj-236", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-222", 0 ],
					"source" : [ "obj-237", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-271", 0 ],
					"source" : [ "obj-238", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-252", 0 ],
					"source" : [ "obj-239", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-28", 0 ],
					"source" : [ "obj-24", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-239", 0 ],
					"order" : 1,
					"source" : [ "obj-240", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-257", 0 ],
					"order" : 0,
					"source" : [ "obj-240", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-238", 2 ],
					"order" : 0,
					"source" : [ "obj-241", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-240", 2 ],
					"order" : 1,
					"source" : [ "obj-241", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-238", 1 ],
					"order" : 0,
					"source" : [ "obj-242", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-240", 1 ],
					"order" : 1,
					"source" : [ "obj-242", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-255", 0 ],
					"source" : [ "obj-243", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-248", 0 ],
					"source" : [ "obj-244", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-244", 0 ],
					"order" : 1,
					"source" : [ "obj-245", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-250", 0 ],
					"order" : 0,
					"source" : [ "obj-245", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-243", 2 ],
					"order" : 0,
					"source" : [ "obj-246", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-245", 2 ],
					"order" : 1,
					"source" : [ "obj-246", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-243", 1 ],
					"order" : 0,
					"source" : [ "obj-247", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-245", 1 ],
					"order" : 1,
					"source" : [ "obj-247", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-243", 0 ],
					"source" : [ "obj-248", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-24", 0 ],
					"order" : 1,
					"source" : [ "obj-25", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-45", 1 ],
					"order" : 0,
					"source" : [ "obj-25", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-249", 0 ],
					"source" : [ "obj-250", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-88", 0 ],
					"source" : [ "obj-251", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-238", 0 ],
					"source" : [ "obj-252", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-245", 0 ],
					"source" : [ "obj-255", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-256", 0 ],
					"source" : [ "obj-257", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-23", 2 ],
					"order" : 0,
					"source" : [ "obj-26", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 2 ],
					"order" : 1,
					"source" : [ "obj-26", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-270", 0 ],
					"source" : [ "obj-260", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-265", 0 ],
					"source" : [ "obj-261", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-261", 0 ],
					"order" : 1,
					"source" : [ "obj-262", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-267", 0 ],
					"order" : 0,
					"source" : [ "obj-262", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-260", 2 ],
					"order" : 0,
					"source" : [ "obj-263", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-262", 2 ],
					"order" : 1,
					"source" : [ "obj-263", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-260", 1 ],
					"order" : 0,
					"source" : [ "obj-264", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-262", 1 ],
					"order" : 1,
					"source" : [ "obj-264", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-260", 0 ],
					"source" : [ "obj-265", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-266", 0 ],
					"source" : [ "obj-267", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-23", 1 ],
					"order" : 0,
					"source" : [ "obj-27", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 1 ],
					"order" : 1,
					"source" : [ "obj-27", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-262", 0 ],
					"source" : [ "obj-270", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-240", 0 ],
					"source" : [ "obj-271", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-284", 0 ],
					"source" : [ "obj-272", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-277", 0 ],
					"source" : [ "obj-273", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-273", 0 ],
					"order" : 1,
					"source" : [ "obj-274", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-281", 0 ],
					"order" : 0,
					"source" : [ "obj-274", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-272", 2 ],
					"order" : 0,
					"source" : [ "obj-275", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-274", 2 ],
					"order" : 1,
					"source" : [ "obj-275", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-272", 1 ],
					"order" : 0,
					"source" : [ "obj-276", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-274", 1 ],
					"order" : 1,
					"source" : [ "obj-276", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-272", 0 ],
					"source" : [ "obj-277", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-280", 0 ],
					"hidden" : 1,
					"source" : [ "obj-278", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-23", 0 ],
					"source" : [ "obj-28", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-207", 0 ],
					"hidden" : 1,
					"source" : [ "obj-280", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-279", 0 ],
					"source" : [ "obj-281", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-274", 0 ],
					"source" : [ "obj-284", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 0 ],
					"source" : [ "obj-29", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-13", 0 ],
					"source" : [ "obj-3", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-29", 0 ],
					"source" : [ "obj-30", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-466", 0 ],
					"source" : [ "obj-303", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4810", 0 ],
					"source" : [ "obj-308", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-65", 0 ],
					"source" : [ "obj-31", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-309", 0 ],
					"source" : [ "obj-310", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-308", 0 ],
					"order" : 1,
					"source" : [ "obj-313", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-310", 0 ],
					"order" : 0,
					"source" : [ "obj-313", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 0 ],
					"source" : [ "obj-32", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-333", 0 ],
					"source" : [ "obj-324", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-328", 0 ],
					"source" : [ "obj-325", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-325", 0 ],
					"order" : 1,
					"source" : [ "obj-326", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-334", 0 ],
					"order" : 0,
					"source" : [ "obj-326", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-324", 1 ],
					"order" : 0,
					"source" : [ "obj-327", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-326", 1 ],
					"order" : 1,
					"source" : [ "obj-327", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-324", 0 ],
					"source" : [ "obj-328", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-52", 0 ],
					"source" : [ "obj-33", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-329", 0 ],
					"source" : [ "obj-330", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-326", 0 ],
					"source" : [ "obj-333", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-330", 0 ],
					"source" : [ "obj-334", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-340", 0 ],
					"source" : [ "obj-335", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-339", 0 ],
					"source" : [ "obj-336", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-334", 1 ],
					"order" : 0,
					"source" : [ "obj-337", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-336", 0 ],
					"order" : 1,
					"source" : [ "obj-337", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-335", 1 ],
					"order" : 0,
					"source" : [ "obj-338", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-337", 1 ],
					"order" : 1,
					"source" : [ "obj-338", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-335", 0 ],
					"source" : [ "obj-339", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-38", 0 ],
					"source" : [ "obj-34", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-337", 0 ],
					"source" : [ "obj-340", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-346", 0 ],
					"source" : [ "obj-341", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-345", 0 ],
					"source" : [ "obj-342", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-334", 2 ],
					"order" : 0,
					"source" : [ "obj-343", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-342", 0 ],
					"order" : 1,
					"source" : [ "obj-343", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-341", 1 ],
					"order" : 0,
					"source" : [ "obj-344", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-343", 1 ],
					"order" : 1,
					"source" : [ "obj-344", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-341", 0 ],
					"source" : [ "obj-345", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-343", 0 ],
					"source" : [ "obj-346", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-356", 0 ],
					"source" : [ "obj-347", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-351", 0 ],
					"source" : [ "obj-348", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-348", 0 ],
					"order" : 1,
					"source" : [ "obj-349", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-353", 0 ],
					"order" : 0,
					"source" : [ "obj-349", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-34", 0 ],
					"order" : 1,
					"source" : [ "obj-35", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-45", 0 ],
					"order" : 0,
					"source" : [ "obj-35", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-347", 1 ],
					"order" : 0,
					"source" : [ "obj-350", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-349", 1 ],
					"order" : 1,
					"source" : [ "obj-350", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-347", 0 ],
					"source" : [ "obj-351", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-352", 0 ],
					"source" : [ "obj-353", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-349", 0 ],
					"source" : [ "obj-356", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-33", 2 ],
					"order" : 0,
					"source" : [ "obj-36", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-35", 2 ],
					"order" : 1,
					"source" : [ "obj-36", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-33", 1 ],
					"order" : 0,
					"source" : [ "obj-37", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-35", 1 ],
					"order" : 1,
					"source" : [ "obj-37", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-33", 0 ],
					"source" : [ "obj-38", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-406", 0 ],
					"source" : [ "obj-397", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-401", 0 ],
					"source" : [ "obj-398", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-398", 0 ],
					"order" : 1,
					"source" : [ "obj-399", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-403", 0 ],
					"order" : 0,
					"source" : [ "obj-399", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 0 ],
					"source" : [ "obj-4", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-42", 0 ],
					"source" : [ "obj-40", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-397", 1 ],
					"order" : 0,
					"source" : [ "obj-400", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-399", 1 ],
					"order" : 1,
					"source" : [ "obj-400", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-397", 0 ],
					"source" : [ "obj-401", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-402", 0 ],
					"source" : [ "obj-403", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-399", 0 ],
					"source" : [ "obj-406", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-416", 0 ],
					"source" : [ "obj-407", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-411", 0 ],
					"source" : [ "obj-408", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-408", 0 ],
					"order" : 1,
					"source" : [ "obj-409", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-413", 0 ],
					"order" : 0,
					"source" : [ "obj-409", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-39", 0 ],
					"source" : [ "obj-41", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-407", 1 ],
					"order" : 0,
					"source" : [ "obj-410", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-409", 1 ],
					"order" : 1,
					"source" : [ "obj-410", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-407", 0 ],
					"source" : [ "obj-411", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-412", 0 ],
					"source" : [ "obj-413", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-409", 0 ],
					"source" : [ "obj-416", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-426", 0 ],
					"source" : [ "obj-417", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-421", 0 ],
					"source" : [ "obj-418", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-418", 0 ],
					"order" : 1,
					"source" : [ "obj-419", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-423", 0 ],
					"order" : 0,
					"source" : [ "obj-419", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-417", 1 ],
					"order" : 0,
					"source" : [ "obj-420", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-419", 1 ],
					"order" : 1,
					"source" : [ "obj-420", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-417", 0 ],
					"source" : [ "obj-421", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-422", 0 ],
					"source" : [ "obj-423", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-419", 0 ],
					"source" : [ "obj-426", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-432", 0 ],
					"source" : [ "obj-427", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-431", 0 ],
					"source" : [ "obj-428", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-428", 0 ],
					"order" : 1,
					"source" : [ "obj-429", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-439", 2 ],
					"order" : 0,
					"source" : [ "obj-429", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-194", 0 ],
					"source" : [ "obj-43", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-427", 1 ],
					"order" : 0,
					"source" : [ "obj-430", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-429", 1 ],
					"order" : 1,
					"source" : [ "obj-430", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-427", 0 ],
					"source" : [ "obj-431", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-429", 0 ],
					"source" : [ "obj-432", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-438", 0 ],
					"source" : [ "obj-433", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-437", 0 ],
					"source" : [ "obj-434", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-434", 0 ],
					"order" : 1,
					"source" : [ "obj-435", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-439", 1 ],
					"order" : 0,
					"source" : [ "obj-435", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-433", 1 ],
					"order" : 0,
					"source" : [ "obj-436", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-435", 1 ],
					"order" : 1,
					"source" : [ "obj-436", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-433", 0 ],
					"source" : [ "obj-437", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-435", 0 ],
					"source" : [ "obj-438", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-446", 0 ],
					"source" : [ "obj-439", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-251", 0 ],
					"source" : [ "obj-44", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-449", 0 ],
					"source" : [ "obj-440", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-444", 0 ],
					"source" : [ "obj-441", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-439", 0 ],
					"order" : 0,
					"source" : [ "obj-442", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-441", 0 ],
					"order" : 1,
					"source" : [ "obj-442", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-440", 1 ],
					"order" : 0,
					"source" : [ "obj-443", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-442", 1 ],
					"order" : 1,
					"source" : [ "obj-443", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-440", 0 ],
					"source" : [ "obj-444", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-445", 0 ],
					"source" : [ "obj-446", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-442", 0 ],
					"source" : [ "obj-449", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-41", 0 ],
					"source" : [ "obj-45", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4812", 0 ],
					"source" : [ "obj-451", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-452", 0 ],
					"source" : [ "obj-453", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-451", 0 ],
					"order" : 1,
					"source" : [ "obj-456", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-453", 0 ],
					"order" : 0,
					"source" : [ "obj-456", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-461", 0 ],
					"source" : [ "obj-457", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-457", 0 ],
					"order" : 1,
					"source" : [ "obj-458", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-463", 0 ],
					"order" : 0,
					"source" : [ "obj-458", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-303", 2 ],
					"order" : 0,
					"source" : [ "obj-459", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-458", 2 ],
					"order" : 1,
					"source" : [ "obj-459", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-303", 1 ],
					"order" : 0,
					"source" : [ "obj-460", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-458", 1 ],
					"order" : 1,
					"source" : [ "obj-460", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-303", 0 ],
					"source" : [ "obj-461", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-462", 0 ],
					"source" : [ "obj-463", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-458", 0 ],
					"source" : [ "obj-466", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-919", 0 ],
					"source" : [ "obj-4685", 3 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-919", 3 ],
					"source" : [ "obj-4685", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-919", 2 ],
					"source" : [ "obj-4685", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-919", 1 ],
					"source" : [ "obj-4685", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4685", 0 ],
					"source" : [ "obj-4686", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4693", 0 ],
					"source" : [ "obj-4687", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4692", 0 ],
					"source" : [ "obj-4688", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4688", 0 ],
					"order" : 1,
					"source" : [ "obj-4689", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4709", 2 ],
					"order" : 0,
					"source" : [ "obj-4689", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-471", 0 ],
					"source" : [ "obj-469", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4687", 2 ],
					"order" : 0,
					"source" : [ "obj-4690", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4689", 2 ],
					"order" : 1,
					"source" : [ "obj-4690", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4687", 1 ],
					"order" : 0,
					"source" : [ "obj-4691", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4689", 1 ],
					"order" : 1,
					"source" : [ "obj-4691", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4687", 0 ],
					"source" : [ "obj-4692", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4689", 0 ],
					"source" : [ "obj-4693", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4700", 0 ],
					"source" : [ "obj-4694", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4699", 0 ],
					"source" : [ "obj-4695", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4695", 0 ],
					"order" : 1,
					"source" : [ "obj-4696", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4709", 1 ],
					"order" : 0,
					"source" : [ "obj-4696", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4694", 2 ],
					"order" : 0,
					"source" : [ "obj-4697", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4696", 2 ],
					"order" : 1,
					"source" : [ "obj-4697", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4694", 1 ],
					"order" : 0,
					"source" : [ "obj-4698", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4696", 1 ],
					"order" : 1,
					"source" : [ "obj-4698", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4694", 0 ],
					"source" : [ "obj-4699", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4696", 0 ],
					"source" : [ "obj-4700", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4712", 0 ],
					"source" : [ "obj-4701", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4706", 0 ],
					"source" : [ "obj-4702", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4702", 0 ],
					"order" : 1,
					"source" : [ "obj-4703", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4709", 0 ],
					"order" : 0,
					"source" : [ "obj-4703", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4701", 2 ],
					"order" : 0,
					"source" : [ "obj-4704", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4703", 2 ],
					"order" : 1,
					"source" : [ "obj-4704", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4701", 1 ],
					"order" : 0,
					"source" : [ "obj-4705", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4703", 1 ],
					"order" : 1,
					"source" : [ "obj-4705", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4701", 0 ],
					"source" : [ "obj-4706", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4707", 0 ],
					"source" : [ "obj-4708", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4686", 0 ],
					"source" : [ "obj-4709", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-470", 0 ],
					"source" : [ "obj-471", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4703", 0 ],
					"source" : [ "obj-4712", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4719", 0 ],
					"source" : [ "obj-4713", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4718", 0 ],
					"source" : [ "obj-4714", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4714", 0 ],
					"order" : 1,
					"source" : [ "obj-4715", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4735", 2 ],
					"order" : 0,
					"source" : [ "obj-4715", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4713", 2 ],
					"order" : 0,
					"source" : [ "obj-4716", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4715", 2 ],
					"order" : 1,
					"source" : [ "obj-4716", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4713", 1 ],
					"order" : 0,
					"source" : [ "obj-4717", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4715", 1 ],
					"order" : 1,
					"source" : [ "obj-4717", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4713", 0 ],
					"source" : [ "obj-4718", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4715", 0 ],
					"source" : [ "obj-4719", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4726", 0 ],
					"source" : [ "obj-4720", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4725", 0 ],
					"source" : [ "obj-4721", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4721", 0 ],
					"order" : 1,
					"source" : [ "obj-4722", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4735", 1 ],
					"order" : 0,
					"source" : [ "obj-4722", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4720", 2 ],
					"order" : 0,
					"source" : [ "obj-4723", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4722", 2 ],
					"order" : 1,
					"source" : [ "obj-4723", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4720", 1 ],
					"order" : 0,
					"source" : [ "obj-4724", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4722", 1 ],
					"order" : 1,
					"source" : [ "obj-4724", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4720", 0 ],
					"source" : [ "obj-4725", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4722", 0 ],
					"source" : [ "obj-4726", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4738", 0 ],
					"source" : [ "obj-4727", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4732", 0 ],
					"source" : [ "obj-4728", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4728", 0 ],
					"order" : 1,
					"source" : [ "obj-4729", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4735", 0 ],
					"order" : 0,
					"source" : [ "obj-4729", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4727", 2 ],
					"order" : 0,
					"source" : [ "obj-4730", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4729", 2 ],
					"order" : 1,
					"source" : [ "obj-4730", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4727", 1 ],
					"order" : 0,
					"source" : [ "obj-4731", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4729", 1 ],
					"order" : 1,
					"source" : [ "obj-4731", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4727", 0 ],
					"source" : [ "obj-4732", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4733", 0 ],
					"source" : [ "obj-4734", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4734", 0 ],
					"source" : [ "obj-4735", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4729", 0 ],
					"source" : [ "obj-4738", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4749", 0 ],
					"source" : [ "obj-4739", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4744", 0 ],
					"source" : [ "obj-4740", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4740", 0 ],
					"order" : 1,
					"source" : [ "obj-4741", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4746", 0 ],
					"order" : 0,
					"source" : [ "obj-4741", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4739", 2 ],
					"order" : 0,
					"source" : [ "obj-4742", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4741", 2 ],
					"order" : 1,
					"source" : [ "obj-4742", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4739", 1 ],
					"order" : 0,
					"source" : [ "obj-4743", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4741", 1 ],
					"order" : 1,
					"source" : [ "obj-4743", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4739", 0 ],
					"source" : [ "obj-4744", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4745", 0 ],
					"source" : [ "obj-4746", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4741", 0 ],
					"source" : [ "obj-4749", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4760", 0 ],
					"source" : [ "obj-4750", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4755", 0 ],
					"source" : [ "obj-4751", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4751", 0 ],
					"order" : 1,
					"source" : [ "obj-4752", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4757", 0 ],
					"order" : 0,
					"source" : [ "obj-4752", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4750", 2 ],
					"order" : 0,
					"source" : [ "obj-4753", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4752", 2 ],
					"order" : 1,
					"source" : [ "obj-4753", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4750", 1 ],
					"order" : 0,
					"source" : [ "obj-4754", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4752", 1 ],
					"order" : 1,
					"source" : [ "obj-4754", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4750", 0 ],
					"source" : [ "obj-4755", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4756", 0 ],
					"source" : [ "obj-4757", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4752", 0 ],
					"source" : [ "obj-4760", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4771", 0 ],
					"source" : [ "obj-4761", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4766", 0 ],
					"source" : [ "obj-4762", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4762", 0 ],
					"order" : 1,
					"source" : [ "obj-4763", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4768", 0 ],
					"order" : 0,
					"source" : [ "obj-4763", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4761", 2 ],
					"order" : 0,
					"source" : [ "obj-4764", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4763", 2 ],
					"order" : 1,
					"source" : [ "obj-4764", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4761", 1 ],
					"order" : 0,
					"source" : [ "obj-4765", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4763", 1 ],
					"order" : 1,
					"source" : [ "obj-4765", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4761", 0 ],
					"source" : [ "obj-4766", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4767", 0 ],
					"source" : [ "obj-4768", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4763", 0 ],
					"source" : [ "obj-4771", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4782", 0 ],
					"source" : [ "obj-4772", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4777", 0 ],
					"source" : [ "obj-4773", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4773", 0 ],
					"order" : 1,
					"source" : [ "obj-4774", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4779", 0 ],
					"order" : 0,
					"source" : [ "obj-4774", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4772", 2 ],
					"order" : 0,
					"source" : [ "obj-4775", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4774", 2 ],
					"order" : 1,
					"source" : [ "obj-4775", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4772", 1 ],
					"order" : 0,
					"source" : [ "obj-4776", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4774", 1 ],
					"order" : 1,
					"source" : [ "obj-4776", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4772", 0 ],
					"source" : [ "obj-4777", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4778", 0 ],
					"source" : [ "obj-4779", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4774", 0 ],
					"source" : [ "obj-4782", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4793", 0 ],
					"source" : [ "obj-4783", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4788", 0 ],
					"source" : [ "obj-4784", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4784", 0 ],
					"order" : 1,
					"source" : [ "obj-4785", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4790", 0 ],
					"order" : 0,
					"source" : [ "obj-4785", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4783", 2 ],
					"order" : 0,
					"source" : [ "obj-4786", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4785", 2 ],
					"order" : 1,
					"source" : [ "obj-4786", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4783", 1 ],
					"order" : 0,
					"source" : [ "obj-4787", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4785", 1 ],
					"order" : 1,
					"source" : [ "obj-4787", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4783", 0 ],
					"source" : [ "obj-4788", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4789", 0 ],
					"source" : [ "obj-4790", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4785", 0 ],
					"source" : [ "obj-4793", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4804", 0 ],
					"source" : [ "obj-4794", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4799", 0 ],
					"source" : [ "obj-4795", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4795", 0 ],
					"order" : 1,
					"source" : [ "obj-4796", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4801", 0 ],
					"order" : 0,
					"source" : [ "obj-4796", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4794", 2 ],
					"order" : 0,
					"source" : [ "obj-4797", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4796", 2 ],
					"order" : 1,
					"source" : [ "obj-4797", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4794", 1 ],
					"order" : 0,
					"source" : [ "obj-4798", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4796", 1 ],
					"order" : 1,
					"source" : [ "obj-4798", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4794", 0 ],
					"source" : [ "obj-4799", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-85", 0 ],
					"source" : [ "obj-48", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4800", 0 ],
					"source" : [ "obj-4801", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4796", 0 ],
					"source" : [ "obj-4804", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4894", 0 ],
					"order" : 0,
					"source" : [ "obj-4806", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4897", 0 ],
					"order" : 1,
					"source" : [ "obj-4806", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4906", 0 ],
					"source" : [ "obj-4807", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4905", 0 ],
					"source" : [ "obj-4808", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4964", 0 ],
					"source" : [ "obj-4809", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-920", 0 ],
					"source" : [ "obj-4810", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4965", 0 ],
					"source" : [ "obj-4811", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-921", 0 ],
					"source" : [ "obj-4812", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4966", 0 ],
					"source" : [ "obj-4813", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4971", 0 ],
					"source" : [ "obj-4815", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4972", 0 ],
					"source" : [ "obj-4816", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4973", 0 ],
					"source" : [ "obj-4817", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4978", 0 ],
					"source" : [ "obj-4818", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4979", 0 ],
					"source" : [ "obj-4819", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4980", 0 ],
					"source" : [ "obj-4820", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4938", 0 ],
					"source" : [ "obj-4821", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4939", 0 ],
					"source" : [ "obj-4822", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4940", 0 ],
					"source" : [ "obj-4823", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4945", 0 ],
					"source" : [ "obj-4824", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4946", 0 ],
					"source" : [ "obj-4825", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4947", 0 ],
					"source" : [ "obj-4826", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4952", 0 ],
					"source" : [ "obj-4827", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4953", 0 ],
					"source" : [ "obj-4828", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4954", 0 ],
					"source" : [ "obj-4829", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4912", 0 ],
					"source" : [ "obj-4830", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4913", 0 ],
					"source" : [ "obj-4831", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4914", 0 ],
					"source" : [ "obj-4832", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4919", 0 ],
					"source" : [ "obj-4833", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4920", 0 ],
					"source" : [ "obj-4834", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4921", 0 ],
					"source" : [ "obj-4835", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4926", 0 ],
					"source" : [ "obj-4836", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4927", 0 ],
					"source" : [ "obj-4837", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4928", 0 ],
					"source" : [ "obj-4838", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5034", 0 ],
					"source" : [ "obj-4851", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5056", 0 ],
					"source" : [ "obj-4854", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5057", 0 ],
					"source" : [ "obj-4855", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5058", 0 ],
					"source" : [ "obj-4856", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5045", 0 ],
					"source" : [ "obj-4857", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5046", 0 ],
					"source" : [ "obj-4858", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5047", 0 ],
					"source" : [ "obj-4859", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5147", 0 ],
					"source" : [ "obj-4897", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4904", 1 ],
					"source" : [ "obj-4898", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4907", 0 ],
					"source" : [ "obj-4899", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-93", 0 ],
					"source" : [ "obj-49", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4915", 0 ],
					"order" : 0,
					"source" : [ "obj-4900", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4922", 0 ],
					"order" : 1,
					"source" : [ "obj-4900", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4934", 0 ],
					"order" : 2,
					"source" : [ "obj-4900", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4900", 0 ],
					"source" : [ "obj-4901", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4906", 0 ],
					"source" : [ "obj-4904", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4904", 0 ],
					"source" : [ "obj-4905", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4899", 0 ],
					"source" : [ "obj-4906", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4908", 0 ],
					"source" : [ "obj-4907", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4898", 0 ],
					"source" : [ "obj-4908", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4901", 0 ],
					"source" : [ "obj-4908", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4915", 0 ],
					"source" : [ "obj-4909", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4914", 0 ],
					"source" : [ "obj-4910", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4910", 0 ],
					"order" : 1,
					"source" : [ "obj-4911", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4931", 2 ],
					"order" : 0,
					"source" : [ "obj-4911", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4909", 2 ],
					"order" : 0,
					"source" : [ "obj-4912", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4911", 2 ],
					"order" : 1,
					"source" : [ "obj-4912", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4909", 1 ],
					"order" : 0,
					"source" : [ "obj-4913", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4911", 1 ],
					"order" : 1,
					"source" : [ "obj-4913", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4909", 0 ],
					"source" : [ "obj-4914", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4911", 0 ],
					"source" : [ "obj-4915", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4922", 0 ],
					"source" : [ "obj-4916", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4921", 0 ],
					"source" : [ "obj-4917", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4917", 0 ],
					"order" : 1,
					"source" : [ "obj-4918", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4931", 1 ],
					"order" : 0,
					"source" : [ "obj-4918", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4916", 2 ],
					"order" : 0,
					"source" : [ "obj-4919", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4918", 2 ],
					"order" : 1,
					"source" : [ "obj-4919", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4916", 1 ],
					"order" : 0,
					"source" : [ "obj-4920", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4918", 1 ],
					"order" : 1,
					"source" : [ "obj-4920", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4916", 0 ],
					"source" : [ "obj-4921", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4918", 0 ],
					"source" : [ "obj-4922", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4934", 0 ],
					"source" : [ "obj-4923", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4928", 0 ],
					"source" : [ "obj-4924", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4924", 0 ],
					"order" : 1,
					"source" : [ "obj-4925", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4931", 0 ],
					"order" : 0,
					"source" : [ "obj-4925", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4923", 2 ],
					"order" : 0,
					"source" : [ "obj-4926", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4925", 2 ],
					"order" : 1,
					"source" : [ "obj-4926", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4923", 1 ],
					"order" : 0,
					"source" : [ "obj-4927", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4925", 1 ],
					"order" : 1,
					"source" : [ "obj-4927", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4923", 0 ],
					"source" : [ "obj-4928", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4929", 0 ],
					"source" : [ "obj-4930", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4930", 0 ],
					"source" : [ "obj-4931", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4925", 0 ],
					"source" : [ "obj-4934", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4941", 0 ],
					"source" : [ "obj-4935", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4940", 0 ],
					"source" : [ "obj-4936", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4936", 0 ],
					"order" : 1,
					"source" : [ "obj-4937", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4957", 2 ],
					"order" : 0,
					"source" : [ "obj-4937", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4935", 2 ],
					"order" : 0,
					"source" : [ "obj-4938", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4937", 2 ],
					"order" : 1,
					"source" : [ "obj-4938", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4935", 1 ],
					"order" : 0,
					"source" : [ "obj-4939", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4937", 1 ],
					"order" : 1,
					"source" : [ "obj-4939", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4935", 0 ],
					"source" : [ "obj-4940", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4937", 0 ],
					"source" : [ "obj-4941", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4948", 0 ],
					"source" : [ "obj-4942", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4947", 0 ],
					"source" : [ "obj-4943", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4943", 0 ],
					"order" : 1,
					"source" : [ "obj-4944", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4957", 1 ],
					"order" : 0,
					"source" : [ "obj-4944", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4942", 2 ],
					"order" : 0,
					"source" : [ "obj-4945", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4944", 2 ],
					"order" : 1,
					"source" : [ "obj-4945", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4942", 1 ],
					"order" : 0,
					"source" : [ "obj-4946", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4944", 1 ],
					"order" : 1,
					"source" : [ "obj-4946", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4942", 0 ],
					"source" : [ "obj-4947", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4944", 0 ],
					"source" : [ "obj-4948", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4960", 0 ],
					"source" : [ "obj-4949", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4954", 0 ],
					"source" : [ "obj-4950", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4950", 0 ],
					"order" : 1,
					"source" : [ "obj-4951", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4957", 0 ],
					"order" : 0,
					"source" : [ "obj-4951", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4949", 2 ],
					"order" : 0,
					"source" : [ "obj-4952", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4951", 2 ],
					"order" : 1,
					"source" : [ "obj-4952", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4949", 1 ],
					"order" : 0,
					"source" : [ "obj-4953", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4951", 1 ],
					"order" : 1,
					"source" : [ "obj-4953", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4949", 0 ],
					"source" : [ "obj-4954", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4955", 0 ],
					"source" : [ "obj-4956", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4956", 0 ],
					"source" : [ "obj-4957", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4951", 0 ],
					"source" : [ "obj-4960", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4967", 0 ],
					"source" : [ "obj-4961", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4966", 0 ],
					"source" : [ "obj-4962", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4962", 0 ],
					"order" : 1,
					"source" : [ "obj-4963", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4983", 2 ],
					"order" : 0,
					"source" : [ "obj-4963", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4961", 2 ],
					"order" : 0,
					"source" : [ "obj-4964", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4963", 2 ],
					"order" : 1,
					"source" : [ "obj-4964", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4961", 1 ],
					"order" : 0,
					"source" : [ "obj-4965", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4963", 1 ],
					"order" : 1,
					"source" : [ "obj-4965", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4961", 0 ],
					"source" : [ "obj-4966", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4963", 0 ],
					"source" : [ "obj-4967", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4974", 0 ],
					"source" : [ "obj-4968", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4973", 0 ],
					"source" : [ "obj-4969", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4969", 0 ],
					"order" : 1,
					"source" : [ "obj-4970", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4983", 1 ],
					"order" : 0,
					"source" : [ "obj-4970", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4968", 2 ],
					"order" : 0,
					"source" : [ "obj-4971", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4970", 2 ],
					"order" : 1,
					"source" : [ "obj-4971", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4968", 1 ],
					"order" : 0,
					"source" : [ "obj-4972", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4970", 1 ],
					"order" : 1,
					"source" : [ "obj-4972", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4968", 0 ],
					"source" : [ "obj-4973", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4970", 0 ],
					"source" : [ "obj-4974", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4986", 0 ],
					"source" : [ "obj-4975", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4980", 0 ],
					"source" : [ "obj-4976", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4976", 0 ],
					"order" : 1,
					"source" : [ "obj-4977", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4983", 0 ],
					"order" : 0,
					"source" : [ "obj-4977", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4975", 2 ],
					"order" : 0,
					"source" : [ "obj-4978", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4977", 2 ],
					"order" : 1,
					"source" : [ "obj-4978", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4975", 1 ],
					"order" : 0,
					"source" : [ "obj-4979", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4977", 1 ],
					"order" : 1,
					"source" : [ "obj-4979", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4975", 0 ],
					"source" : [ "obj-4980", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4981", 0 ],
					"source" : [ "obj-4982", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4982", 0 ],
					"source" : [ "obj-4983", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4977", 0 ],
					"source" : [ "obj-4986", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4997", 0 ],
					"source" : [ "obj-4987", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4992", 0 ],
					"source" : [ "obj-4988", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4988", 0 ],
					"order" : 1,
					"source" : [ "obj-4989", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4994", 0 ],
					"order" : 0,
					"source" : [ "obj-4989", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4987", 2 ],
					"order" : 0,
					"source" : [ "obj-4990", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4989", 2 ],
					"order" : 1,
					"source" : [ "obj-4990", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4987", 1 ],
					"order" : 0,
					"source" : [ "obj-4991", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4989", 1 ],
					"order" : 1,
					"source" : [ "obj-4991", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4987", 0 ],
					"source" : [ "obj-4992", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4993", 0 ],
					"source" : [ "obj-4994", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4989", 0 ],
					"source" : [ "obj-4997", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5008", 0 ],
					"source" : [ "obj-4998", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5003", 0 ],
					"source" : [ "obj-4999", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 0 ],
					"order" : 0,
					"source" : [ "obj-5", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4", 0 ],
					"order" : 1,
					"source" : [ "obj-5", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-51", 1 ],
					"hidden" : 1,
					"source" : [ "obj-50", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-54", 0 ],
					"hidden" : 1,
					"source" : [ "obj-50", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4999", 0 ],
					"order" : 1,
					"source" : [ "obj-5000", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5005", 0 ],
					"order" : 0,
					"source" : [ "obj-5000", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4998", 2 ],
					"order" : 0,
					"source" : [ "obj-5001", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5000", 2 ],
					"order" : 1,
					"source" : [ "obj-5001", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4998", 1 ],
					"order" : 0,
					"source" : [ "obj-5002", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5000", 1 ],
					"order" : 1,
					"source" : [ "obj-5002", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4998", 0 ],
					"source" : [ "obj-5003", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5004", 0 ],
					"source" : [ "obj-5005", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5000", 0 ],
					"source" : [ "obj-5008", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5019", 0 ],
					"source" : [ "obj-5009", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5014", 0 ],
					"source" : [ "obj-5010", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5010", 0 ],
					"order" : 1,
					"source" : [ "obj-5011", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5016", 0 ],
					"order" : 0,
					"source" : [ "obj-5011", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5009", 2 ],
					"order" : 0,
					"source" : [ "obj-5012", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5011", 2 ],
					"order" : 1,
					"source" : [ "obj-5012", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5009", 1 ],
					"order" : 0,
					"source" : [ "obj-5013", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5011", 1 ],
					"order" : 1,
					"source" : [ "obj-5013", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5009", 0 ],
					"source" : [ "obj-5014", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5015", 0 ],
					"source" : [ "obj-5016", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5011", 0 ],
					"source" : [ "obj-5019", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5030", 0 ],
					"source" : [ "obj-5020", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5025", 0 ],
					"source" : [ "obj-5021", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5021", 0 ],
					"order" : 1,
					"source" : [ "obj-5022", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5027", 0 ],
					"order" : 0,
					"source" : [ "obj-5022", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5020", 2 ],
					"order" : 0,
					"source" : [ "obj-5023", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5022", 2 ],
					"order" : 1,
					"source" : [ "obj-5023", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5020", 1 ],
					"order" : 0,
					"source" : [ "obj-5024", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5022", 1 ],
					"order" : 1,
					"source" : [ "obj-5024", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5020", 0 ],
					"source" : [ "obj-5025", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5026", 0 ],
					"source" : [ "obj-5027", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5022", 0 ],
					"source" : [ "obj-5030", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5041", 0 ],
					"source" : [ "obj-5031", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5036", 0 ],
					"source" : [ "obj-5032", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5032", 0 ],
					"order" : 1,
					"source" : [ "obj-5033", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5038", 0 ],
					"order" : 0,
					"source" : [ "obj-5033", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5031", 2 ],
					"order" : 0,
					"source" : [ "obj-5034", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5033", 2 ],
					"order" : 1,
					"source" : [ "obj-5034", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5031", 1 ],
					"order" : 0,
					"source" : [ "obj-5035", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5033", 1 ],
					"order" : 1,
					"source" : [ "obj-5035", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5031", 0 ],
					"source" : [ "obj-5036", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5037", 0 ],
					"source" : [ "obj-5038", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5033", 0 ],
					"source" : [ "obj-5041", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5052", 0 ],
					"source" : [ "obj-5042", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5047", 0 ],
					"source" : [ "obj-5043", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5043", 0 ],
					"order" : 1,
					"source" : [ "obj-5044", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5049", 0 ],
					"order" : 0,
					"source" : [ "obj-5044", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5042", 2 ],
					"order" : 0,
					"source" : [ "obj-5045", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5044", 2 ],
					"order" : 1,
					"source" : [ "obj-5045", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5042", 1 ],
					"order" : 0,
					"source" : [ "obj-5046", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5044", 1 ],
					"order" : 1,
					"source" : [ "obj-5046", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5042", 0 ],
					"source" : [ "obj-5047", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5048", 0 ],
					"source" : [ "obj-5049", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5044", 0 ],
					"source" : [ "obj-5052", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5063", 0 ],
					"source" : [ "obj-5053", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5058", 0 ],
					"source" : [ "obj-5054", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5054", 0 ],
					"order" : 1,
					"source" : [ "obj-5055", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5060", 0 ],
					"order" : 0,
					"source" : [ "obj-5055", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5053", 2 ],
					"order" : 0,
					"source" : [ "obj-5056", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5055", 2 ],
					"order" : 1,
					"source" : [ "obj-5056", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5053", 1 ],
					"order" : 0,
					"source" : [ "obj-5057", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5055", 1 ],
					"order" : 1,
					"source" : [ "obj-5057", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5053", 0 ],
					"source" : [ "obj-5058", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5059", 0 ],
					"source" : [ "obj-5060", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5055", 0 ],
					"source" : [ "obj-5063", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5065", 0 ],
					"source" : [ "obj-5064", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4863", 0 ],
					"order" : 0,
					"source" : [ "obj-5065", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5066", 0 ],
					"order" : 1,
					"source" : [ "obj-5065", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5064", 0 ],
					"order" : 1,
					"source" : [ "obj-5066", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5074", 2 ],
					"order" : 0,
					"source" : [ "obj-5066", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5068", 0 ],
					"source" : [ "obj-5067", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4864", 0 ],
					"order" : 0,
					"source" : [ "obj-5068", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5069", 0 ],
					"order" : 1,
					"source" : [ "obj-5068", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5067", 0 ],
					"order" : 1,
					"source" : [ "obj-5069", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5074", 1 ],
					"order" : 0,
					"source" : [ "obj-5069", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5071", 0 ],
					"source" : [ "obj-5070", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4865", 0 ],
					"order" : 0,
					"source" : [ "obj-5071", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5077", 0 ],
					"order" : 1,
					"source" : [ "obj-5071", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5072", 0 ],
					"source" : [ "obj-5073", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5073", 0 ],
					"source" : [ "obj-5074", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5070", 0 ],
					"order" : 1,
					"source" : [ "obj-5077", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5074", 0 ],
					"order" : 0,
					"source" : [ "obj-5077", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5079", 0 ],
					"source" : [ "obj-5078", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4860", 0 ],
					"order" : 0,
					"source" : [ "obj-5079", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5080", 0 ],
					"order" : 1,
					"source" : [ "obj-5079", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5078", 0 ],
					"order" : 1,
					"source" : [ "obj-5080", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5088", 2 ],
					"order" : 0,
					"source" : [ "obj-5080", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5082", 0 ],
					"source" : [ "obj-5081", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4861", 0 ],
					"order" : 0,
					"source" : [ "obj-5082", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5083", 0 ],
					"order" : 1,
					"source" : [ "obj-5082", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5081", 0 ],
					"order" : 1,
					"source" : [ "obj-5083", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5088", 1 ],
					"order" : 0,
					"source" : [ "obj-5083", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5085", 0 ],
					"source" : [ "obj-5084", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4862", 0 ],
					"order" : 0,
					"source" : [ "obj-5085", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5091", 0 ],
					"order" : 1,
					"source" : [ "obj-5085", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5086", 0 ],
					"source" : [ "obj-5087", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5087", 0 ],
					"source" : [ "obj-5088", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5084", 0 ],
					"order" : 1,
					"source" : [ "obj-5091", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5088", 0 ],
					"order" : 0,
					"source" : [ "obj-5091", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5108", 0 ],
					"source" : [ "obj-5099", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-58", 0 ],
					"hidden" : 1,
					"source" : [ "obj-51", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5103", 0 ],
					"source" : [ "obj-5100", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5100", 0 ],
					"order" : 1,
					"source" : [ "obj-5101", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5105", 0 ],
					"order" : 0,
					"source" : [ "obj-5101", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4892", 0 ],
					"order" : 0,
					"source" : [ "obj-5102", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5099", 1 ],
					"order" : 1,
					"source" : [ "obj-5102", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5101", 1 ],
					"order" : 2,
					"source" : [ "obj-5102", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4893", 0 ],
					"order" : 0,
					"source" : [ "obj-5103", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5099", 0 ],
					"order" : 1,
					"source" : [ "obj-5103", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5104", 0 ],
					"source" : [ "obj-5105", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5101", 0 ],
					"source" : [ "obj-5108", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5118", 0 ],
					"source" : [ "obj-5109", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5113", 0 ],
					"source" : [ "obj-5110", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5110", 0 ],
					"order" : 1,
					"source" : [ "obj-5111", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5115", 0 ],
					"order" : 0,
					"source" : [ "obj-5111", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4890", 0 ],
					"order" : 0,
					"source" : [ "obj-5112", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5109", 1 ],
					"order" : 1,
					"source" : [ "obj-5112", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5111", 1 ],
					"order" : 2,
					"source" : [ "obj-5112", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4891", 0 ],
					"order" : 0,
					"source" : [ "obj-5113", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5109", 0 ],
					"order" : 1,
					"source" : [ "obj-5113", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5114", 0 ],
					"source" : [ "obj-5115", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5111", 0 ],
					"source" : [ "obj-5118", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5124", 0 ],
					"source" : [ "obj-5119", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5123", 0 ],
					"source" : [ "obj-5120", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5120", 0 ],
					"order" : 1,
					"source" : [ "obj-5121", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5131", 2 ],
					"order" : 0,
					"source" : [ "obj-5121", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4884", 0 ],
					"order" : 0,
					"source" : [ "obj-5122", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5119", 1 ],
					"order" : 1,
					"source" : [ "obj-5122", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5121", 1 ],
					"order" : 2,
					"source" : [ "obj-5122", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4885", 0 ],
					"order" : 0,
					"source" : [ "obj-5123", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5119", 0 ],
					"order" : 1,
					"source" : [ "obj-5123", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5121", 0 ],
					"source" : [ "obj-5124", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5130", 0 ],
					"source" : [ "obj-5125", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5129", 0 ],
					"source" : [ "obj-5126", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5126", 0 ],
					"order" : 1,
					"source" : [ "obj-5127", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5131", 1 ],
					"order" : 0,
					"source" : [ "obj-5127", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4886", 0 ],
					"order" : 0,
					"source" : [ "obj-5128", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5125", 1 ],
					"order" : 1,
					"source" : [ "obj-5128", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5127", 1 ],
					"order" : 2,
					"source" : [ "obj-5128", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4887", 0 ],
					"order" : 0,
					"source" : [ "obj-5129", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5125", 0 ],
					"order" : 1,
					"source" : [ "obj-5129", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5127", 0 ],
					"source" : [ "obj-5130", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5138", 0 ],
					"source" : [ "obj-5131", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5141", 0 ],
					"source" : [ "obj-5132", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5136", 0 ],
					"source" : [ "obj-5133", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5131", 0 ],
					"order" : 0,
					"source" : [ "obj-5134", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5133", 0 ],
					"order" : 1,
					"source" : [ "obj-5134", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4888", 0 ],
					"order" : 0,
					"source" : [ "obj-5135", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5132", 1 ],
					"order" : 1,
					"source" : [ "obj-5135", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5134", 1 ],
					"order" : 2,
					"source" : [ "obj-5135", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4889", 0 ],
					"order" : 0,
					"source" : [ "obj-5136", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5132", 0 ],
					"order" : 1,
					"source" : [ "obj-5136", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5137", 0 ],
					"source" : [ "obj-5138", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5134", 0 ],
					"source" : [ "obj-5141", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4806", 0 ],
					"source" : [ "obj-5142", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5143", 0 ],
					"source" : [ "obj-5144", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5142", 0 ],
					"order" : 1,
					"source" : [ "obj-5147", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5144", 0 ],
					"order" : 0,
					"source" : [ "obj-5147", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-35", 0 ],
					"source" : [ "obj-52", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-123", 0 ],
					"source" : [ "obj-53", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-207", 0 ],
					"source" : [ "obj-54", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-118", 0 ],
					"source" : [ "obj-55", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-120", 0 ],
					"order" : 0,
					"source" : [ "obj-56", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 0 ],
					"order" : 1,
					"source" : [ "obj-56", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-53", 2 ],
					"order" : 0,
					"source" : [ "obj-57", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-56", 2 ],
					"order" : 1,
					"source" : [ "obj-57", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-634", 0 ],
					"source" : [ "obj-577", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1004", 0 ],
					"source" : [ "obj-579", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-207", 0 ],
					"source" : [ "obj-58", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-60", 0 ],
					"source" : [ "obj-59", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-62", 0 ],
					"source" : [ "obj-60", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-607", 0 ],
					"source" : [ "obj-606", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-608", 0 ],
					"source" : [ "obj-607", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-606", 0 ],
					"order" : 1,
					"source" : [ "obj-608", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-616", 2 ],
					"order" : 0,
					"source" : [ "obj-608", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-610", 0 ],
					"source" : [ "obj-609", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-85", 0 ],
					"hidden" : 1,
					"source" : [ "obj-61", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-611", 0 ],
					"source" : [ "obj-610", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-609", 0 ],
					"order" : 1,
					"source" : [ "obj-611", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-616", 1 ],
					"order" : 0,
					"source" : [ "obj-611", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-613", 0 ],
					"source" : [ "obj-612", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-619", 0 ],
					"source" : [ "obj-613", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-614", 0 ],
					"source" : [ "obj-615", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-615", 0 ],
					"source" : [ "obj-616", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-612", 0 ],
					"order" : 1,
					"source" : [ "obj-619", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-616", 0 ],
					"order" : 0,
					"source" : [ "obj-619", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-59", 0 ],
					"order" : 1,
					"source" : [ "obj-62", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-77", 2 ],
					"order" : 0,
					"source" : [ "obj-62", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-621", 0 ],
					"source" : [ "obj-620", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-622", 0 ],
					"source" : [ "obj-621", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-620", 0 ],
					"order" : 1,
					"source" : [ "obj-622", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-630", 2 ],
					"order" : 0,
					"source" : [ "obj-622", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-624", 0 ],
					"source" : [ "obj-623", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-625", 0 ],
					"source" : [ "obj-624", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-623", 0 ],
					"order" : 1,
					"source" : [ "obj-625", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-630", 1 ],
					"order" : 0,
					"source" : [ "obj-625", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-627", 0 ],
					"source" : [ "obj-626", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-633", 0 ],
					"source" : [ "obj-627", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-628", 0 ],
					"source" : [ "obj-629", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-66", 0 ],
					"source" : [ "obj-63", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-629", 0 ],
					"source" : [ "obj-630", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-626", 0 ],
					"order" : 1,
					"source" : [ "obj-633", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-630", 0 ],
					"order" : 0,
					"source" : [ "obj-633", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-635", 0 ],
					"source" : [ "obj-634", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-577", 0 ],
					"order" : 1,
					"source" : [ "obj-635", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-640", 1 ],
					"order" : 0,
					"source" : [ "obj-635", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-637", 0 ],
					"source" : [ "obj-636", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-643", 0 ],
					"source" : [ "obj-637", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-638", 0 ],
					"source" : [ "obj-639", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-31", 0 ],
					"source" : [ "obj-64", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-639", 0 ],
					"source" : [ "obj-640", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-636", 0 ],
					"order" : 1,
					"source" : [ "obj-643", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-640", 0 ],
					"order" : 0,
					"source" : [ "obj-643", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-645", 0 ],
					"source" : [ "obj-644", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-646", 0 ],
					"source" : [ "obj-645", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-644", 0 ],
					"order" : 1,
					"source" : [ "obj-646", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-654", 2 ],
					"order" : 0,
					"source" : [ "obj-646", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-648", 0 ],
					"source" : [ "obj-647", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-649", 0 ],
					"source" : [ "obj-648", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-647", 0 ],
					"order" : 1,
					"source" : [ "obj-649", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-654", 1 ],
					"order" : 0,
					"source" : [ "obj-649", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 0 ],
					"order" : 0,
					"source" : [ "obj-65", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-64", 0 ],
					"order" : 1,
					"source" : [ "obj-65", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-651", 0 ],
					"source" : [ "obj-650", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-657", 0 ],
					"source" : [ "obj-651", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-652", 0 ],
					"source" : [ "obj-653", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-653", 0 ],
					"source" : [ "obj-654", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-650", 0 ],
					"order" : 1,
					"source" : [ "obj-657", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-654", 0 ],
					"order" : 0,
					"source" : [ "obj-657", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-668", 0 ],
					"source" : [ "obj-658", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-663", 0 ],
					"source" : [ "obj-659", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-69", 0 ],
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-659", 0 ],
					"order" : 1,
					"source" : [ "obj-660", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-665", 0 ],
					"order" : 0,
					"source" : [ "obj-660", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-658", 2 ],
					"order" : 0,
					"source" : [ "obj-661", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-660", 2 ],
					"order" : 1,
					"source" : [ "obj-661", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-658", 1 ],
					"order" : 0,
					"source" : [ "obj-662", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-660", 1 ],
					"order" : 1,
					"source" : [ "obj-662", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-658", 0 ],
					"source" : [ "obj-663", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-664", 0 ],
					"source" : [ "obj-665", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-660", 0 ],
					"source" : [ "obj-668", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-679", 0 ],
					"source" : [ "obj-669", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-68", 0 ],
					"source" : [ "obj-67", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-674", 0 ],
					"source" : [ "obj-670", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-670", 0 ],
					"order" : 1,
					"source" : [ "obj-671", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-676", 0 ],
					"order" : 0,
					"source" : [ "obj-671", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-669", 2 ],
					"order" : 0,
					"source" : [ "obj-672", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-671", 2 ],
					"order" : 1,
					"source" : [ "obj-672", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-669", 1 ],
					"order" : 0,
					"source" : [ "obj-673", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-671", 1 ],
					"order" : 1,
					"source" : [ "obj-673", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-669", 0 ],
					"source" : [ "obj-674", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-675", 0 ],
					"source" : [ "obj-676", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-671", 0 ],
					"source" : [ "obj-679", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-70", 0 ],
					"source" : [ "obj-68", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-690", 0 ],
					"source" : [ "obj-680", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-685", 0 ],
					"source" : [ "obj-681", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-681", 0 ],
					"order" : 1,
					"source" : [ "obj-682", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-687", 0 ],
					"order" : 0,
					"source" : [ "obj-682", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-680", 2 ],
					"order" : 0,
					"source" : [ "obj-683", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-682", 2 ],
					"order" : 1,
					"source" : [ "obj-683", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-680", 1 ],
					"order" : 0,
					"source" : [ "obj-684", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-682", 1 ],
					"order" : 1,
					"source" : [ "obj-684", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-680", 0 ],
					"source" : [ "obj-685", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-686", 0 ],
					"source" : [ "obj-687", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-63", 0 ],
					"order" : 1,
					"source" : [ "obj-69", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-77", 1 ],
					"order" : 0,
					"source" : [ "obj-69", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-682", 0 ],
					"source" : [ "obj-690", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-701", 0 ],
					"source" : [ "obj-691", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-696", 0 ],
					"source" : [ "obj-692", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-692", 0 ],
					"order" : 1,
					"source" : [ "obj-693", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-698", 0 ],
					"order" : 0,
					"source" : [ "obj-693", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-691", 2 ],
					"order" : 0,
					"source" : [ "obj-694", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-693", 2 ],
					"order" : 1,
					"source" : [ "obj-694", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-691", 1 ],
					"order" : 0,
					"source" : [ "obj-695", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-693", 1 ],
					"order" : 1,
					"source" : [ "obj-695", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-691", 0 ],
					"source" : [ "obj-696", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-697", 0 ],
					"source" : [ "obj-698", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 1 ],
					"order" : 0,
					"source" : [ "obj-7", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 1 ],
					"order" : 1,
					"source" : [ "obj-7", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 1 ],
					"order" : 0,
					"source" : [ "obj-70", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-67", 0 ],
					"order" : 1,
					"source" : [ "obj-70", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-693", 0 ],
					"source" : [ "obj-701", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-712", 0 ],
					"source" : [ "obj-702", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-707", 0 ],
					"source" : [ "obj-703", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-703", 0 ],
					"order" : 1,
					"source" : [ "obj-704", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-709", 0 ],
					"order" : 0,
					"source" : [ "obj-704", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-702", 2 ],
					"order" : 0,
					"source" : [ "obj-705", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-704", 2 ],
					"order" : 1,
					"source" : [ "obj-705", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-702", 1 ],
					"order" : 0,
					"source" : [ "obj-706", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-704", 1 ],
					"order" : 1,
					"source" : [ "obj-706", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-702", 0 ],
					"source" : [ "obj-707", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-708", 0 ],
					"source" : [ "obj-709", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-74", 0 ],
					"source" : [ "obj-71", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-704", 0 ],
					"source" : [ "obj-712", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-723", 0 ],
					"source" : [ "obj-713", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-718", 0 ],
					"source" : [ "obj-714", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-714", 0 ],
					"order" : 1,
					"source" : [ "obj-715", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-720", 0 ],
					"order" : 0,
					"source" : [ "obj-715", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-713", 2 ],
					"order" : 0,
					"source" : [ "obj-716", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-715", 2 ],
					"order" : 1,
					"source" : [ "obj-716", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-713", 1 ],
					"order" : 0,
					"source" : [ "obj-717", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-715", 1 ],
					"order" : 1,
					"source" : [ "obj-717", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-713", 0 ],
					"source" : [ "obj-718", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-719", 0 ],
					"source" : [ "obj-720", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-715", 0 ],
					"source" : [ "obj-723", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-734", 0 ],
					"source" : [ "obj-724", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-729", 0 ],
					"source" : [ "obj-725", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-725", 0 ],
					"order" : 1,
					"source" : [ "obj-726", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-731", 0 ],
					"order" : 0,
					"source" : [ "obj-726", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-724", 2 ],
					"order" : 0,
					"source" : [ "obj-727", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-726", 2 ],
					"order" : 1,
					"source" : [ "obj-727", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-724", 1 ],
					"order" : 0,
					"source" : [ "obj-728", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-726", 1 ],
					"order" : 1,
					"source" : [ "obj-728", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-724", 0 ],
					"source" : [ "obj-729", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-730", 0 ],
					"source" : [ "obj-731", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-726", 0 ],
					"source" : [ "obj-734", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-80", 0 ],
					"source" : [ "obj-74", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-75", 0 ],
					"source" : [ "obj-76", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-76", 0 ],
					"source" : [ "obj-77", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"source" : [ "obj-8", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-71", 0 ],
					"order" : 1,
					"source" : [ "obj-80", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-77", 0 ],
					"order" : 0,
					"source" : [ "obj-80", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-50", 0 ],
					"hidden" : 1,
					"order" : 0,
					"source" : [ "obj-84", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-61", 0 ],
					"hidden" : 1,
					"order" : 1,
					"source" : [ "obj-84", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-51", 0 ],
					"hidden" : 1,
					"source" : [ "obj-85", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 0 ],
					"source" : [ "obj-88", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4708", 0 ],
					"source" : [ "obj-919", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-313", 0 ],
					"source" : [ "obj-920", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-456", 0 ],
					"source" : [ "obj-921", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-48", 0 ],
					"source" : [ "obj-93", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-96", 0 ],
					"source" : [ "obj-95", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-97", 0 ],
					"source" : [ "obj-96", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 2 ],
					"order" : 0,
					"source" : [ "obj-97", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-95", 0 ],
					"order" : 1,
					"source" : [ "obj-97", 0 ]
				}

			}
 ],
		"dependency_cache" : [ 			{
				"name" : "controls.json",
				"bootpath" : "~/Documents/Publications/Papers/SoundSphere_xCoAx2025/Software/Raymarching/RayMarching_Python_v2/controls",
				"patcherrelativepath" : ".",
				"type" : "JSON",
				"implicit" : 1
			}
 ],
		"autosave" : 0
	}

}
