-- identify cases where scan duplicated a face within an image
SELECT ph.file_path, fd.bounding_box, count(*) AS copies
FROM face_detection fd JOIN photo ph ON ph.id = fd.photo_id
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
           ORDER BY (person_id IS NOT NULL) DESC,
                    (suggested_person_id IS NOT NULL) DESC,
                    id ASC
         ) AS rn
  FROM face_detection
)
SELECT id, person_id FROM ranked WHERE rn > 1;

DELETE FROM face_detection WHERE id IN (SELECT id FROM to_del);

-- face_count is not stored (see person_v); only the timestamp needs touching.
UPDATE person p
SET updated_at = now()
WHERE p.id IN (SELECT DISTINCT person_id FROM to_del WHERE person_id IS NOT NULL);
COMMIT;


