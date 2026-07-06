-- identify cases where scan duplicated a face within an image
SELECT ph.file_path, fd.bounding_box, count(*) AS copies
FROM face_detections fd JOIN photos ph ON ph.id = fd.photo_id
GROUP BY ph.file_path, fd.bounding_box
HAVING count(*) > 1
ORDER BY copies DESC;


-- delete duplicates from above
BEGIN;
CREATE TEMP TABLE to_del ON COMMIT DROP AS
WITH ranked AS (
  SELECT id, person_id,
         ROW_NUMBER() OVER (
           PARTITION BY photo_id, bounding_box
           ORDER BY is_validated DESC, (person_id IS NOT NULL) DESC,
                    (suggested_person_id IS NOT NULL) DESC,
                    (blur_score IS NOT NULL) DESC, id ASC
         ) AS rn
  FROM face_detections
)
SELECT id, person_id FROM ranked WHERE rn > 1;

DELETE FROM face_detections WHERE id IN (SELECT id FROM to_del);

UPDATE persons p
SET face_count = (SELECT count(*) FROM face_detections fd WHERE fd.person_id = p.id),
    updated_at = now()
WHERE p.id IN (SELECT DISTINCT person_id FROM to_del WHERE person_id IS NOT NULL);
COMMIT;


