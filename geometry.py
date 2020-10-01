import numpy as np
import torch
import sys
from torch.nn import functional as F
import util


def compute_normal_map(x_img, y_img, z, intrinsics):
    cam_coords = lift(x_img, y_img, z, intrinsics)
    cam_coords = util.lin2img(cam_coords)

    shift_left = cam_coords[:, :, 2:, :]
    shift_right = cam_coords[:, :, :-2, :]

    shift_up = cam_coords[:, :, :, 2:]
    shift_down = cam_coords[:, :, :, :-2]

    diff_hor = F.normalize(shift_right - shift_left, dim=1)[:, :, :, 1:-1]
    diff_ver = F.normalize(shift_up - shift_down, dim=1)[:, :, 1:-1, :]

    cross = torch.cross(diff_hor, diff_ver, dim=1)
    return cross


def get_ray_directions_cam(uv, intrinsics):
    '''Translates meshgrid of uv pixel coordinates to normalized directions of rays through these pixels,
    in camera coordinates.
    '''
    batch_size, num_samples, _ = uv.shape

    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = torch.ones((batch_size, num_samples)).cuda()

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=False)  # (batch_size, -1, 4)
    ray_dirs = F.normalize(pixel_points_cam, dim=2)
    return ray_dirs


def reflect_vector_on_vector(vector_to_reflect, reflection_axis):
    refl = F.normalize(vector_to_reflect.cuda())
    ax = F.normalize(reflection_axis.cuda())

    r = 2 * (ax * refl).sum(dim=1, keepdim=True) * ax - refl
    return r


def parse_intrinsics(intrinsics):
    return 140, 140, 64, 64


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - torch.ones_like(x)*cx) / (torch.ones_like(x)*fx) * z
    y_lift = (y - torch.ones_like(y)*cy) / (torch.ones_like(y)*fy) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def project(x, y, z, intrinsics):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_proj = expand_as(fx, x) * x / z + expand_as(cx, x)
    y_proj = expand_as(fy, y) * y / z + expand_as(cy, y)

    return torch.stack((x_proj, y_proj, z), dim=-1)


def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates plus depth to  world coordinates.
    '''
    batch_size, _, _ = cam2world.shape

    x_cam = xy[:, :, 0].view(batch_size, -1)
    y_cam = xy[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)*-1

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)  # (batch_size, -1, 4)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(cam2world, pixel_points_cam).permute(0, 2, 1)[:, :, :3]  # (batch_size, -1, 3)

    # print(world_coords[0][11])
    # print(xy[0][11])
    # print(depth[0][11])
    # print(cam2world[0])
    # print(depth.size(), cam2world.size(), world_coords[0].size())

    # print(world_from_depth2(xy[0][11], depth[0][11].item(), cam2world[0].cpu()))
    # print(world_from_depth3(xy[0][11], depth[0][11].item(), cam2world[0].cpu()))

    return world_coords

def world_from_depth3(pixel_coord, depth, ext_mat):
    fx, fy, cx, cy = 140.0, 140.0, 64.0, 64.0
    x, y = pixel_coord
    z = -depth
    x_lift = (x - cx) / fx * z
    y_lift = (y - cy) / fy * z
    return torch.mm(torch.tensor(ext_mat), torch.tensor([x_lift, y_lift, -depth, 1]).unsqueeze(1))

def world_from_depth2(pixel_coord, depth, ext_mat):
    p_x, p_y = pixel_coord
    im_x, im_y = p_x/128, (128-p_y)/128
    frame = [
          [
            -0.09142857044935226,
            -0.09142857044935226,
            0.20000000298023224
          ],
          [
            -0.09142857044935226,
            0.09142857044935226,
            0.20000000298023224
          ],
          [
            0.09142857044935226,
            0.09142857044935226,
            0.20000000298023224
          ]
        ]
    frame = [(np.array(v) / (v[2] / depth)) for v in frame]
    min_x, max_x = frame[1][0], frame[2][0]
    min_y, max_y = frame[0][1], frame[1][1]

    cx = im_x * (max_x - min_x) + min_x
    cy = im_y * (max_y - min_y) + min_y
    world = torch.mm(torch.tensor(ext_mat), torch.tensor([cx, cy, -depth, 1]).unsqueeze(1))
    return world.squeeze()

def project_point_on_line(projection_point, line_direction, point_on_line, dim):
    '''Projects a batch of points on a batch of lines as defined by their direction and a point on each line. '''
    assert torch.allclose((line_direction ** 2).sum(dim=dim, keepdim=True).cuda(), torch.Tensor([1]).cuda())
    return point_on_line + ((projection_point - point_on_line) * line_direction).sum(dim=dim,
                                                                                     keepdim=True) * line_direction

def get_ray_directions(xy, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates to normalized directions of rays through these pixels.
    '''
    batch_size, num_samples, _ = xy.shape

    z_cam = torch.ones((batch_size, num_samples)).cuda()
    pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world)  # (batch, num_samples, 3)

    cam_pos = cam2world[:, :3, 3]
    ray_dirs = pixel_points - cam_pos[:, None, :]  # (batch, num_samples, 3)
    ray_dirs = F.normalize(ray_dirs, dim=2)
    return ray_dirs


def depth_from_world(world_coords, cam2world):
    batch_size, num_samples, _ = world_coords.shape

    points_hom = torch.cat((world_coords, torch.ones((batch_size, num_samples, 1)).cuda()),
                           dim=2)  # (batch, num_samples, 4)

    # permute for bmm
    points_hom = points_hom.permute(0, 2, 1)

    points_cam = torch.inverse(cam2world).bmm(points_hom)  # (batch, 4, num_samples)
    depth = points_cam[:, 2, :][:, :, None]  # (batch, num_samples, 1)
    return depth


if __name__ == '__main__':
    import json
    with open('/home/sontung/thesis/photorealistic-blocksworld/blocks-4-3/scene/CLEVR_new_000000.json') as json_file:
        data = json.load(json_file)
    print(data["objects"][0]["location"])
    print(data["objects"][0]["pixel_coords"])
    print(world_from_depth2((35, 87), 14.700079917907715, data["objects"][0]["recon_data"]["matrix_world"]))
    print(world_from_depth3((35, 87), 14.700079917907715, data["objects"][0]["recon_data"]["matrix_world"]))

